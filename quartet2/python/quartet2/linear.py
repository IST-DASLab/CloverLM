import torch
from flashinfer import mm_fp4

from scipy.linalg import hadamard
from .quant import NVFP4Quant, quant_fp4, quant_gridflip_fp4, rht128_quant_eden, rht128_requant, NVFP4QuantMode
import nvtx
import contextlib
from typing import Literal

FP4MatmulBackend = Literal["flashinfer", "qutlass", "dequantized"]
FP4_MATMUL_BACKENDS = ("flashinfer", "qutlass", "dequantized")
_fp4_mm_backend: FP4MatmulBackend = "flashinfer"
FP4WeightQuantizer = Literal["four_six", "gridflip"]
FP4_WEIGHT_QUANTIZERS = ("four_six", "gridflip")
_fp4_weight_quantizer: FP4WeightQuantizer = "four_six"
_gridflip_shift: float = 0.25


def _import_qutlass():
    try:
        import qutlass
    except ImportError as exc:
        raise ImportError(
            "The qutlass FP4 matmul backend requires the qutlass package. "
            "Install the local GridFlip/QuTLASS package before selecting "
            "quartet2.linear.set_fp4_mm_backend('qutlass')."
        ) from exc
    return qutlass

def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
        )


def nvtx_annotate(name: str, color: str = "green"):
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    else:
        return nvtx.annotate(name, color=color)


def rerotate_hadamard(hadamard_matrix):
    signs = torch.randint(
            0, 2, (hadamard_matrix.size(0),),
            device=hadamard_matrix.device,
            dtype=hadamard_matrix.dtype
        ) * 2 - 1
    return hadamard_matrix * signs[None, :] # NOTE: rerotate along last dim, inner dim for TN GEMM


def set_fp4_mm_backend(backend: FP4MatmulBackend) -> FP4MatmulBackend:
    if backend not in FP4_MATMUL_BACKENDS:
        raise ValueError(f"backend must be one of {FP4_MATMUL_BACKENDS}, got {backend!r}")
    if backend == "qutlass":
        _import_qutlass()

    global _fp4_mm_backend
    old_backend = _fp4_mm_backend
    _fp4_mm_backend = backend
    return old_backend


def get_fp4_mm_backend() -> FP4MatmulBackend:
    return _fp4_mm_backend


def set_fp4_weight_quantizer(
        quantizer: FP4WeightQuantizer,
        *,
        gridflip_shift: float = 0.25,
) -> FP4WeightQuantizer:
    if quantizer not in FP4_WEIGHT_QUANTIZERS:
        raise ValueError(f"quantizer must be one of {FP4_WEIGHT_QUANTIZERS}, got {quantizer!r}")
    if gridflip_shift < 0:
        raise ValueError("gridflip_shift must be non-negative")

    global _fp4_weight_quantizer, _gridflip_shift
    old_quantizer = _fp4_weight_quantizer
    _fp4_weight_quantizer = quantizer
    _gridflip_shift = float(gridflip_shift)
    return old_quantizer


def get_fp4_weight_quantizer() -> FP4WeightQuantizer:
    return _fp4_weight_quantizer


def get_gridflip_shift() -> float:
    return _gridflip_shift


@contextlib.contextmanager
def fp4_mm_backend(backend: FP4MatmulBackend):
    old_backend = set_fp4_mm_backend(backend)
    try:
        yield
    finally:
        set_fp4_mm_backend(old_backend)


def _resolve_fp4_mm_backend(backend: FP4MatmulBackend | None) -> FP4MatmulBackend:
    return _fp4_mm_backend if backend is None else backend


@torch.compiler.disable()
def apply_block_transform(x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    if transform is None:
        return x

    rows, cols = x.shape
    group_size = transform.shape[-1]
    groups = cols // group_size
    x_grouped = x.reshape(rows, groups, group_size).permute(1, 0, 2)
    y_grouped = torch.bmm(x_grouped, transform.to(dtype=x.dtype))
    return y_grouped.permute(1, 0, 2).reshape(rows, cols).contiguous()


@torch.compiler.disable()
def update_wush_moments(
        sigma_x: torch.Tensor,
        sigma_w: torch.Tensor | None,
        ema_count: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_size: int,
        ema_decay: float,
) -> None:
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"WUSH requires input dimension divisible by {group_size}, got {x.shape[-1]}")

    rows = x.shape[0]
    out = weight.shape[0]
    groups = sigma_x.shape[0]
    beta = 0.0 if ema_count.item() == 0 else ema_decay

    x_blocks = x.to(torch.float32).reshape(rows, groups, group_size).permute(1, 2, 0)
    sig_x = torch.bmm(x_blocks, x_blocks.transpose(-1, -2)) / max(rows, 1)
    sigma_x.mul_(beta).add_(sig_x, alpha=1.0 - beta)

    if sigma_w is not None:
        w_blocks = weight.detach().to(torch.float32).reshape(out, groups, group_size).permute(1, 2, 0)
        sig_w = torch.bmm(w_blocks, w_blocks.transpose(-1, -2)) / max(out, 1)
        sigma_w.mul_(beta).add_(sig_w, alpha=1.0 - beta)

    ema_count.add_(1)


@torch.compiler.disable()
def _psd_factor_from_moments(
        sigma: torch.Tensor,
        damp: float,
        eye_batch: torch.Tensor,
        eig_floor: float = 1e-8,
) -> torch.Tensor:
    sigma = sigma.float()
    sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1e4, neginf=-1e4)
    damped = sigma + damp * eye_batch

    factor, info = torch.linalg.cholesky_ex(damped)
    bad = info != 0
    if not bad.any():
        return factor

    try:
        eigvals, eigvecs = torch.linalg.eigh(damped[bad])
        eigvals_sqrt = eigvals.clamp_min(eig_floor).sqrt()
        factor[bad] = eigvecs @ torch.diag_embed(eigvals_sqrt) @ eigvecs.transpose(-1, -2)
    except RuntimeError:
        factor[bad] = eye_batch[bad]
    return factor


@torch.compiler.disable()
def get_wush_transforms_from_moments(
        sigma_x: torch.Tensor,
        sigma_w: torch.Tensor | None,
        group_size: int,
        damp: float,
        s_min: float,
        max_cond: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = sigma_x.shape[0]
    device = sigma_x.device
    had = get_hadamard_matrix(group_size, torch.float32, device)
    eye = torch.eye(group_size, device=device, dtype=torch.float32)
    eye_batch = eye.unsqueeze(0).expand(groups, -1, -1)

    if sigma_w is None:
        w_prime = eye_batch.clone()
    else:
        w_prime = _psd_factor_from_moments(sigma_w, damp, eye_batch)

    x_prime = _psd_factor_from_moments(sigma_x, damp, eye_batch)
    _, singular_values, vt = torch.linalg.svd(w_prime.transpose(-1, -2) @ x_prime)
    singular_values = singular_values.clamp_min(s_min)
    s_inv_sqrt = torch.diag_embed(singular_values.rsqrt())

    t_w = had.unsqueeze(0) @ s_inv_sqrt @ vt @ x_prime.transpose(-1, -2)
    t_w_inv, inv_info = torch.linalg.inv_ex(t_w)

    try:
        conds = torch.linalg.cond(t_w)
    except RuntimeError:
        conds = torch.full((groups,), float("inf"), device=device, dtype=torch.float32)
    bad = (inv_info != 0) | (conds > max_cond) | ~torch.isfinite(conds)
    if bad.any():
        t_w[bad] = had
        t_w_inv[bad] = had.T

    # Quartet-II forward applies x @ input_transform and weight @ weight_transform.
    input_transform = t_w_inv
    weight_transform = t_w.transpose(-1, -2)
    return input_transform.to(torch.bfloat16), weight_transform.to(torch.bfloat16), conds


@torch.library.custom_op("quartet2::fp4_mm_flashinfer", mutates_args=())
def _fp4_mm_flashinfer(x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    m, packed_k = x_fp4.shape
    k = packed_k * 2
    n = w_fp4.shape[0]
    BLOCK = 16
    out = torch.empty([m, n], device=x_fp4.device, dtype=torch.bfloat16)

    mm_fp4(
        x_fp4,
        w_fp4.T,
        x_mx.view(-1, k // BLOCK),
        w_mx.view(-1, k // BLOCK).T,
        alpha,
        torch.bfloat16,
        out,
        block_size=BLOCK,
        use_8x4_sf_layout=False,
        backend="cudnn",
        use_nvfp4=True,
    )

    return out


@_fp4_mm_flashinfer.register_fake
def _fp4_mm_flashinfer_fake(x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.empty((x_fp4.shape[0], w_fp4.shape[0]), device=x_fp4.device, dtype=torch.bfloat16)


@torch.library.custom_op("quartet2::fp4_mm_qutlass", mutates_args=())
def _fp4_mm_qutlass(x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    qutlass = _import_qutlass()

    # quartet2 quantization kernels already write micro-scales in Blackwell's
    # blocked layout, even though tensors keep the logical [rows, K / 16] shape.
    return qutlass.matmul_nvf4_bf16_tn(
        x_fp4.contiguous(),
        w_fp4.contiguous(),
        x_mx.contiguous(),
        w_mx.contiguous(),
        alpha.reshape(1).contiguous(),
    )


def _fp4_mm_gridflip_qutlass(
        x_fp4: torch.Tensor,
        w_fp4: torch.Tensor,
        x_mx: torch.Tensor,
        w_mx_rowmajor: torch.Tensor,
        alpha: torch.Tensor,
        grid_shift: float,
) -> torch.Tensor:
    qutlass = _import_qutlass()
    x_mx_rowmajor = unblock(x_mx, x_fp4.shape[0], x_fp4.shape[1] * 2).contiguous()
    return qutlass.matmul_nvf4_gridflip_bf16_tn(
        x_fp4.contiguous(),
        w_fp4.contiguous(),
        x_mx_rowmajor,
        w_mx_rowmajor.contiguous(),
        alpha.reshape(1).contiguous(),
        grid_shift,
    )


@_fp4_mm_qutlass.register_fake
def _fp4_mm_qutlass_fake(x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.empty((x_fp4.shape[0], w_fp4.shape[0]), device=x_fp4.device, dtype=torch.bfloat16)


def to_blocked(input_matrix) -> torch.Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.

    This function is copied from qutlass, but compatible with torch.compile.

    See:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)

    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """
    rows, cols = input_matrix.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    # Note: No second argument to assert, that broke torch.compile
    assert (rows, cols) == (padded_rows, padded_cols)

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()

def unblock(blocked_scales, rows, cols):
    n_row_blocks, n_col_blocks = rows // 128, (cols // 16) // 4
    rearranged = blocked_scales.reshape(-1, 32, 4, 4)
    rearranged = rearranged.permute(0, 2, 1, 3).reshape(-1, n_col_blocks, 128, 4)
    rearranged = rearranged.permute(0, 2, 1, 3)
    # Reverse: view(n_row_blocks, 128, n_col_blocks, 4)
    return rearranged.reshape(n_row_blocks * 128, n_col_blocks * 4)


@torch.compile
def _dq_fp4(x_e2m1: torch.Tensor, x_e4m3: torch.Tensor, alpha: float):
    device = x_e2m1.device

    x_e2m1_i32 = x_e2m1.view(dtype=torch.uint8).to(dtype=torch.int32)
    x_e2m1_unpacked = torch.stack(
        [x_e2m1_i32 & 0xF, (x_e2m1_i32 >> 4) & 0xF], dim=-1
    ).flatten(start_dim=-2)

    grid_dq = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    x_fp4_dq = grid_dq[x_e2m1_unpacked]

    scales_dq = x_e4m3.to(torch.float32)
    scales_dq = unblock(scales_dq, x_e2m1.shape[0], x_e2m1.shape[1] * 2)
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq[..., None]).flatten(
        start_dim=-2
    ) * alpha
    return x_dq.to(torch.bfloat16)


def _round_fp4_values_and_codes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_abs = x.abs()
    magnitude = torch.where(
        x_abs >= 5.0,
        torch.full_like(x, 6.0),
        torch.where(
            x_abs >= 3.5,
            torch.full_like(x, 4.0),
            torch.where(
                x_abs >= 2.5,
                torch.full_like(x, 3.0),
                torch.where(
                    x_abs >= 1.75,
                    torch.full_like(x, 2.0),
                    torch.where(
                        x_abs >= 1.25,
                        torch.full_like(x, 1.5),
                        torch.where(
                            x_abs >= 0.75,
                            torch.full_like(x, 1.0),
                            torch.where(
                                x_abs >= 0.25,
                                torch.full_like(x, 0.5),
                                torch.zeros_like(x),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    magnitude_code = torch.where(
        x_abs >= 5.0,
        torch.full(x.shape, 7, device=x.device, dtype=torch.uint8),
        torch.where(
            x_abs >= 3.5,
            torch.full(x.shape, 6, device=x.device, dtype=torch.uint8),
            torch.where(
                x_abs >= 2.5,
                torch.full(x.shape, 5, device=x.device, dtype=torch.uint8),
                torch.where(
                    x_abs >= 1.75,
                    torch.full(x.shape, 4, device=x.device, dtype=torch.uint8),
                    torch.where(
                        x_abs >= 1.25,
                        torch.full(x.shape, 3, device=x.device, dtype=torch.uint8),
                        torch.where(
                            x_abs >= 0.75,
                            torch.full(x.shape, 2, device=x.device, dtype=torch.uint8),
                            torch.where(
                                x_abs >= 0.25,
                                torch.ones(x.shape, device=x.device, dtype=torch.uint8),
                                torch.zeros(x.shape, device=x.device, dtype=torch.uint8),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    sign = x < 0
    values = torch.where(sign, -magnitude, magnitude)
    codes = magnitude_code | sign.to(torch.uint8) * 0x8
    return values, codes


def _pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.uint8)
    return ((codes[..., 1::2] & 0xF) << 4 | (codes[..., ::2] & 0xF)).flatten(start_dim=-2)


@torch.compiler.disable()
def _quant_gridflip_weight_fp4(
        weight: torch.Tensor,
        *,
        amax: torch.Tensor,
        scale_override: float,
        grid_shift: float,
        mode: NVFP4QuantMode,
) -> NVFP4Quant:
    if mode != NVFP4QuantMode.FOUR_SIX:
        raise ValueError("GridFlip weight quantization currently supports only FOUR_SIX mode")
    if weight.dtype != torch.bfloat16:
        raise TypeError("GridFlip weight quantization requires bfloat16 weights")
    if weight.dim() != 2 or weight.shape[0] % 128 != 0 or weight.shape[1] % 128 != 0:
        raise ValueError("GridFlip weight quantization requires a 2D [rows, cols] tensor with both dimensions divisible by 128")
    return quant_gridflip_fp4(
        weight,
        amax=amax,
        scale_override=scale_override,
        grid_shift=grid_shift,
    )


@torch.compiler.disable()
def _quant_gridflip_weight_fp4_reference(
        weight: torch.Tensor,
        *,
        amax: torch.Tensor,
        scale_override: float,
        grid_shift: float,
        mode: NVFP4QuantMode,
) -> NVFP4Quant:
    if mode != NVFP4QuantMode.FOUR_SIX:
        raise ValueError("GridFlip weight quantization currently supports only FOUR_SIX mode")
    if weight.dtype != torch.bfloat16:
        raise TypeError("GridFlip weight quantization requires bfloat16 weights")
    if weight.dim() != 2 or weight.shape[0] % 128 != 0 or weight.shape[1] % 128 != 0:
        raise ValueError("GridFlip weight quantization requires a 2D [rows, cols] tensor with both dimensions divisible by 128")
    standard = quant_fp4(
        weight,
        amax=amax,
        scale_override=scale_override,
        mode=mode,
    )
    rows, cols = weight.shape
    k_blocks = cols // 16
    blocks = weight.float().reshape(rows, k_blocks, 16)
    global_scale = standard.tensor_scale.float().reshape(())

    standard_scale_rowmajor = unblock(
        standard.micro_scales,
        rows,
        cols,
    ).contiguous()
    standard_codes = standard.fp4.view(torch.uint8).to(torch.int32)
    standard_codes = torch.stack(
        [standard_codes & 0xF, (standard_codes >> 4) & 0xF],
        dim=-1,
    ).flatten(start_dim=-2)
    grid_dq = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=weight.device,
    )
    standard_dequant = (
        grid_dq[standard_codes]
        .reshape(rows, k_blocks, 16)
        .mul(standard_scale_rowmajor.float().unsqueeze(-1))
        .mul(global_scale)
    )
    standard_error = (
        standard_dequant.reshape(rows, k_blocks, 16) - blocks
    ).square().sum(dim=-1)

    best_error = torch.full((rows, k_blocks), float("inf"), device=weight.device)
    best_codes = torch.empty((rows, k_blocks, 16), device=weight.device, dtype=torch.uint8)
    best_scale = torch.empty((rows, k_blocks), device=weight.device, dtype=torch.float8_e4m3fn)
    block_abs_max = blocks.abs().amax(dim=-1, keepdim=True)

    for candidate in (6.0, 4.0):
        shifted_val_max = candidate + grid_shift
        raw_scale = block_abs_max * scale_override / (shifted_val_max * global_scale)
        scale_fp8 = raw_scale.to(torch.float8_e4m3fn)
        scale_dq_f32 = scale_fp8.float()
        scale_quant_f32 = torch.where(scale_dq_f32 == 0.0, torch.ones_like(scale_dq_f32), scale_dq_f32)
        scaled = -blocks / (scale_quant_f32 * global_scale) - grid_shift
        q_values, q_codes = _round_fp4_values_and_codes(scaled)
        dequant = -(q_values + grid_shift) * scale_dq_f32 * global_scale
        error = (dequant - blocks).square().sum(dim=-1)
        improve = error < best_error
        best_error = torch.where(improve, error, best_error)
        best_codes = torch.where(improve[..., None], q_codes, best_codes)
        best_scale = torch.where(improve, scale_fp8.reshape(rows, k_blocks), best_scale)

    flags = best_error < standard_error
    shifted_fp4 = _pack_fp4_codes(best_codes)
    fp4 = torch.where(flags.repeat_interleave(8, dim=1), shifted_fp4, standard.fp4)

    scale_rowmajor = torch.where(flags, best_scale, standard_scale_rowmajor)
    scale_u8 = scale_rowmajor.view(torch.uint8)
    scale_u8 |= flags.to(torch.uint8) * 0x80
    return NVFP4Quant(fp4.contiguous(), scale_rowmajor.contiguous(), standard.tensor_scale)


def _dq_gridflip_weight_fp4(
        w_fp4: torch.Tensor,
        w_mx_rowmajor: torch.Tensor,
        grid_shift: float,
) -> torch.Tensor:
    scale_u8 = w_mx_rowmajor.view(torch.uint8)
    flags = (scale_u8 & 0x80).bool()
    clean_scale_u8 = scale_u8 & 0x7F
    clean_scale_rowmajor = clean_scale_u8.view(torch.float8_e4m3fn)
    clean_scale_blocked = to_blocked(clean_scale_rowmajor).view_as(w_mx_rowmajor)
    standard = _dq_fp4(w_fp4, clean_scale_blocked, 1.0).float()
    correction = clean_scale_rowmajor.float().repeat_interleave(16, dim=1)
    flags = flags.repeat_interleave(16, dim=1)
    return torch.where(flags, -standard - grid_shift * correction, standard)


def _fp4_mm_dequantized(x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    x = _dq_fp4(x_fp4, x_mx, 1.0).to(torch.float32)
    w = _dq_fp4(w_fp4, w_mx, 1.0).to(torch.float32)
    return (x @ w.T * alpha.reshape(())).to(torch.bfloat16)


def _fp4_mm_gridflip_dequantized(
        x_fp4: torch.Tensor,
        w_fp4: torch.Tensor,
        x_mx: torch.Tensor,
        w_mx_rowmajor: torch.Tensor,
        alpha: torch.Tensor,
        grid_shift: float,
) -> torch.Tensor:
    x = _dq_fp4(x_fp4, x_mx, 1.0).float()
    w = _dq_gridflip_weight_fp4(w_fp4, w_mx_rowmajor, grid_shift).float()
    return (x @ w.T * alpha.reshape(())).to(torch.bfloat16)


def _fp4_mm_gridflip(
        x_fp4: torch.Tensor,
        w_fp4: torch.Tensor,
        x_mx: torch.Tensor,
        w_mx_rowmajor: torch.Tensor,
        alpha: torch.Tensor,
        *,
        grid_shift: float,
        backend: FP4MatmulBackend | None = None,
) -> torch.Tensor:
    backend = _resolve_fp4_mm_backend(backend)
    if backend == "qutlass":
        return _fp4_mm_gridflip_qutlass(
            x_fp4,
            w_fp4,
            x_mx,
            w_mx_rowmajor,
            alpha,
            grid_shift,
        )
    if backend == "dequantized":
        return _fp4_mm_gridflip_dequantized(
            x_fp4,
            w_fp4,
            x_mx,
            w_mx_rowmajor,
            alpha,
            grid_shift,
        )
    raise ValueError("GridFlip weight quantization requires the qutlass or dequantized FP4 matmul backend")


def _fp4_mm(
        x_fp4: torch.Tensor,
        w_fp4: torch.Tensor,
        x_mx: torch.Tensor,
        w_mx: torch.Tensor,
        alpha: torch.Tensor,
        *,
        backend: FP4MatmulBackend | None = None,
) -> torch.Tensor:
    backend = _resolve_fp4_mm_backend(backend)
    if backend == "flashinfer":
        return _fp4_mm_flashinfer(x_fp4, w_fp4, x_mx, w_mx, alpha)
    if backend == "qutlass":
        return _fp4_mm_qutlass(x_fp4, w_fp4, x_mx, w_mx, alpha)
    if backend == "dequantized":
        return _fp4_mm_dequantized(x_fp4, w_fp4, x_mx, w_mx, alpha)
    raise ValueError(f"backend must be one of {FP4_MATMUL_BACKENDS}, got {backend!r}")


@torch.compile(dynamic=False)
def abs_max(x):
    return x.abs().max().to(torch.float32)

class Quartet_II_fn(torch.autograd.Function):
    group_size = 16

    #@torch.compile(dynamic=False)
    @staticmethod
    def forward(ctx, input, weight, had, mode: NVFP4QuantMode, disable_backward_quant: bool = False,
                weight_amax: torch.Tensor = None, input_amax: torch.Tensor = None,
                scratch_amax: torch.Tensor = None, wush_input_transform: torch.Tensor = None,
                wush_weight_transform: torch.Tensor = None):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        ctx.disable_backward_quant = disable_backward_quant
        ctx.scratch_amax = scratch_amax
        ctx.had = had
        ctx.wush_input_transform = wush_input_transform
        ctx.wush_weight_transform = wush_weight_transform
        ctx.fp4_mm_backend = get_fp4_mm_backend()

        autocast_enabled = torch.is_autocast_enabled("cuda")
        if autocast_enabled:
            input = input.to(torch.bfloat16)
            weight = weight.to(torch.bfloat16)
        elif weight.dtype != torch.bfloat16:
            raise TypeError("Weight must be bfloat16. Either set `dtype=torch.bfloat16` or enable autocast`")
        elif input.dtype != torch.bfloat16:
            raise TypeError("Input must be bfloat16. Either cast input to bfloat16 or enable autocast`")

        forward_scale_override = 1.0

        flat_input = input.reshape(-1, input.shape[-1])
        if wush_input_transform is not None:
            flat_input = apply_block_transform(flat_input.contiguous(), wush_input_transform)
            weight = apply_block_transform(weight.contiguous(), wush_weight_transform)

        with nvtx_annotate("Abs-max", color="red"):
            if input_amax is None:
                input_amax = abs_max(flat_input)
            if weight_amax is None:
                weight_amax = abs_max(weight)

        with nvtx_annotate("Quant", color="yellow"):
            input_fp4 = quant_fp4(flat_input, amax=input_amax, scale_override=forward_scale_override, mode=mode)
            weight_backward_fp4 = quant_fp4(weight, amax=weight_amax, scale_override=forward_scale_override, mode=mode)
            use_gridflip_weight = get_fp4_weight_quantizer() == "gridflip"
            if use_gridflip_weight:
                weight_fp4 = _quant_gridflip_weight_fp4(
                    weight,
                    amax=weight_amax,
                    scale_override=forward_scale_override,
                    grid_shift=get_gridflip_shift(),
                    mode=mode,
                )
            else:
                weight_fp4 = weight_backward_fp4
        ctx.save_for_backward(input_fp4.fp4, input_fp4.micro_scales, input_fp4.tensor_scale,
                              weight_backward_fp4.fp4, weight_backward_fp4.micro_scales, weight_backward_fp4.tensor_scale)
        with nvtx_annotate("Matmul", color="blue"):
            if use_gridflip_weight:
                res = _fp4_mm_gridflip(
                    input_fp4.fp4,
                    weight_fp4.fp4,
                    input_fp4.micro_scales,
                    weight_fp4.micro_scales,
                    alpha=input_fp4.tensor_scale * weight_fp4.tensor_scale,
                    grid_shift=get_gridflip_shift(),
                    backend=ctx.fp4_mm_backend,
                )
            else:
                res = _fp4_mm(
                    input_fp4.fp4, weight_fp4.fp4,
                    input_fp4.micro_scales, weight_fp4.micro_scales,
                    alpha=input_fp4.tensor_scale * weight_fp4.tensor_scale,
                    backend=ctx.fp4_mm_backend)

        return res.reshape(ctx.batch, ctx.seq, ctx.out_dim)

    #@torch.compile(dynamic=False)
    @staticmethod
    def backward(ctx, grad_output):
        # Load ctx and reshape
        xfp4, xs, xm, wfp4, ws, wm = ctx.saved_tensors
        backward_scale_override = (17 / 16) * 0.93

        autocast_enabled = torch.is_autocast_enabled("cuda")
        if autocast_enabled:
            grad_output = grad_output.to(torch.bfloat16)

        # Re-randomize the rotation
        had = rerotate_hadamard(ctx.had)
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.disable_backward_quant:
            xr = _dq_fp4(xfp4, xs, xm)
            wr = _dq_fp4(wfp4, ws, wm)
            grad_input = flat_grad_output @ wr
            grad_weight = flat_grad_output.T @ xr
            if ctx.wush_input_transform is not None:
                grad_input = apply_block_transform(grad_input.contiguous(), ctx.wush_input_transform.transpose(-1, -2))
                grad_weight = apply_block_transform(grad_weight.contiguous(), ctx.wush_weight_transform.transpose(-1, -2))
            return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None, None, None

        # EW
        with nvtx_annotate("Quant", color="yellow"):
            e_ht_fp4, e_ht_ms, e_ht_ts = rht128_quant_eden(x=flat_grad_output, h=had[:16, :], scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
            wt_ht_fp4, wt_ht_ms, wt_ht_ts = rht128_requant(x=wfp4, x_group_scales=ws, x_tensor_scale=wm, h=had[:16, :], scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
        with nvtx_annotate("Matmul", color="blue"):
            grad_input = _fp4_mm(
                e_ht_fp4,
                wt_ht_fp4,
                e_ht_ms,
                wt_ht_ms,
                alpha=e_ht_ts*wt_ht_ts,
                backend=ctx.fp4_mm_backend,
            )

        # EtX
        with nvtx_annotate("Quant", color="yellow"):
            et_ht_fp4, et_ht_ms, et_ht_ts = rht128_quant_eden(x=flat_grad_output, h=had[:16, :], scale_override=backward_scale_override, transpose=True, scratch_amax=ctx.scratch_amax)
            xt_ht_fp4, xt_ht_ms, xt_ht_ts = rht128_requant(x=xfp4, x_group_scales=xs, x_tensor_scale=xm, h=had[:16, :], scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
        with nvtx_annotate("Matmul", color="blue"):
            grad_weight = _fp4_mm(
                et_ht_fp4,
                xt_ht_fp4,
                et_ht_ms,
                xt_ht_ms,
                alpha=et_ht_ts*xt_ht_ts,
                backend=ctx.fp4_mm_backend,
            )
        if ctx.wush_input_transform is not None:
            grad_input = apply_block_transform(grad_input.contiguous(), ctx.wush_input_transform.transpose(-1, -2))
            grad_weight = apply_block_transform(grad_weight.contiguous(), ctx.wush_weight_transform.transpose(-1, -2))
        return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None, None, None


class Quartet_II_linear(torch.nn.Linear):
    def __init__(self, *args, four_over_six=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = NVFP4QuantMode.FOUR_SIX if four_over_six else NVFP4QuantMode.RNE
        self.weight_abs_max = None
        self.wush_enabled = False
        self.wush_update_freq = 200
        self.wush_damp = 1e-3
        self.wush_s_min = 1e-2
        self.wush_max_cond = 1e4
        self.wush_ema_decay = 0.99
        self.wush_group_size = 128
        self.wush_g_identity = True
        self.wush_step = -1
        self.wush_last_update_step = -1
        # initialize hadamard matrix.
        # *if* we are on meta device, initialization will be deferred until we move to real device (handled in _apply)
        had = get_hadamard_matrix(128, torch.bfloat16, self.weight.device) if self.weight.device.type != 'meta' else None
        self.register_buffer("had", had, persistent=False)
        self.register_buffer("wush_input_transform", None, persistent=False)
        self.register_buffer("wush_weight_transform", None, persistent=False)
        self.register_buffer("wush_sigma_x", None, persistent=False)
        self.register_buffer("wush_sigma_w", None, persistent=False)
        self.register_buffer("wush_ema_count", torch.zeros((), dtype=torch.long, device=self.weight.device), persistent=False)
        self.register_buffer("wush_last_conds", None, persistent=False)
        self.register_buffer("scratch_amax", torch.empty((), dtype=torch.uint32, device=self.weight.device), persistent=False)

    def _apply(self, fn):
        old_device = self.weight.device
        super()._apply(fn)
        # if we move from meta device to real device, we need to create the hadamard matrix
        if old_device.type == 'meta' and self.weight.device.type != 'meta':
            self.had = get_hadamard_matrix(128, torch.bfloat16, self.weight.device)
        return self

    @torch.no_grad()
    def configure_wush(self, enabled: bool, update_freq: int = 200, damp: float = 1e-3,
                       s_min: float = 1e-2, max_cond: float = 1e4,
                       ema_decay: float = 0.99, group_size: int = 128,
                       g_identity: bool = True):
        self.wush_enabled = enabled
        self.wush_update_freq = update_freq
        self.wush_damp = damp
        self.wush_s_min = s_min
        self.wush_max_cond = max_cond
        self.wush_ema_decay = ema_decay
        self.wush_group_size = group_size
        self.wush_g_identity = g_identity
        self.wush_last_update_step = -1
        if not enabled:
            self.wush_input_transform = None
            self.wush_weight_transform = None
            self.wush_sigma_x = None
            self.wush_sigma_w = None
            self.wush_ema_count.zero_()
            self.wush_last_conds = None
            return

        if self.weight.shape[1] % group_size != 0:
            raise ValueError(f"WUSH requires in_features divisible by {group_size}, got {self.weight.shape[1]}")

        groups = self.weight.shape[1] // group_size
        device = self.weight.device
        had = get_hadamard_matrix(group_size, torch.bfloat16, device)
        self.wush_input_transform = had.T.unsqueeze(0).expand(groups, -1, -1).clone()
        self.wush_weight_transform = had.T.unsqueeze(0).expand(groups, -1, -1).clone()
        self.wush_sigma_x = torch.zeros((groups, group_size, group_size), device=device, dtype=torch.float32)
        self.wush_sigma_w = None if g_identity else torch.zeros_like(self.wush_sigma_x)
        self.wush_ema_count.zero_()
        self.wush_last_conds = torch.ones((groups,), device=device, dtype=torch.float32)

    def set_wush_step(self, step: int):
        self.wush_step = step

    @torch.compiler.disable()
    @torch.no_grad()
    def update_wush_moments(self, x: torch.Tensor):
        if not self.wush_enabled or not self.training:
            return

        x = x.detach().reshape(-1, x.shape[-1]).contiguous()
        update_wush_moments(
            self.wush_sigma_x,
            self.wush_sigma_w,
            self.wush_ema_count,
            x,
            self.weight.detach(),
            self.wush_group_size,
            self.wush_ema_decay,
        )

    @torch.compiler.disable()
    @torch.no_grad()
    def recompute_wush_transform(self):
        if not self.wush_enabled or self.wush_ema_count.item() == 0:
            return False

        input_transform, weight_transform, conds = get_wush_transforms_from_moments(
            self.wush_sigma_x,
            self.wush_sigma_w,
            self.wush_group_size,
            self.wush_damp,
            self.wush_s_min,
            self.wush_max_cond,
        )
        self.wush_input_transform = input_transform
        self.wush_weight_transform = weight_transform
        self.wush_last_conds = conds.to(dtype=torch.float32)
        return True

    def forward(self, x, disable_backward_quant=False, input_abs_max=None):
        if self.wush_enabled:
            self.update_wush_moments(x)
        return Quartet_II_fn.apply(x, self.weight[...], self.had, self.mode, disable_backward_quant, self.weight_abs_max, input_abs_max, self.scratch_amax, self.wush_input_transform, self.wush_weight_transform)


def register_optimizer_hook(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    opt_params = {id(p) for group in optimizer.param_groups for p in group['params']}
    quartet_modules = [
        m for m in model.modules()
        if isinstance(m, Quartet_II_linear) and id(m.weight) in opt_params
    ]

    def hook(opt, args, kwargs):
        for m in quartet_modules:
            m.weight_abs_max = abs_max(m.weight)

    return optimizer.register_step_post_hook(hook)


def configure_wush(model: torch.nn.Module, enabled: bool, update_freq: int = 200, damp: float = 1e-3,
                   s_min: float = 1e-2, max_cond: float = 1e4, ema_decay: float = 0.99,
                   group_size: int = 128, g_identity: bool = True):
    for module in model.modules():
        if isinstance(module, Quartet_II_linear):
            module.configure_wush(enabled, update_freq, damp, s_min, max_cond, ema_decay, group_size, g_identity)


def set_wush_step(model: torch.nn.Module, step: int):
    for module in model.modules():
        if isinstance(module, Quartet_II_linear):
            module.set_wush_step(step)


@torch.no_grad()
def update_wush_transforms(model: torch.nn.Module, sync_distributed: bool = False):
    modules = [module for module in model.modules() if isinstance(module, Quartet_II_linear) and module.wush_enabled]
    if not modules:
        return 0

    if sync_distributed and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        for module in modules:
            torch.distributed.all_reduce(module.wush_sigma_x, op=torch.distributed.ReduceOp.SUM)
            module.wush_sigma_x.div_(world_size)

    updated = 0
    for module in modules:
        updated += int(module.recompute_wush_transform())

    if sync_distributed and torch.distributed.is_initialized():
        for module in modules:
            for attr in ("wush_input_transform", "wush_weight_transform"):
                tensor = getattr(module, attr)
                flat = tensor.data.clone().contiguous()
                torch.distributed.broadcast(flat, src=0)
                tensor.data.copy_(flat)

    return updated
