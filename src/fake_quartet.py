
from random import randint

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from scipy.linalg import hadamard



def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )


def rerotate_hadamard(hadamard_matrix):
    signs = torch.diag(
        torch.randint(
            0, 2, (hadamard_matrix.size(0),),
            device=hadamard_matrix.device,
            dtype=hadamard_matrix.dtype,
        ) * 2 - 1
    )
    return hadamard_matrix @ signs



@triton.jit
def _rtn_fp4(x):
    x_abs = tl.abs(x)
    x_sign = tl.where(x > 0, 1, -1)
    x_fp4_abs = tl.where(
        x_abs >= 5, 6,
        tl.where(x_abs >= 3.5, 4,
        tl.where(x_abs >= 2.5, 3,
        tl.where(x_abs >= 1.75, 2,
        tl.where(x_abs >= 1.25, 1.5,
        tl.where(x_abs >= 0.75, 1,
        tl.where(x_abs >= 0.25, 0.5,
        0.0)))))))
    return x_fp4_abs * x_sign


@triton.jit
def _get_scales(x, amax, val_max, scales_max):
    s_dec = tl.where(amax == 0.0, 1.0, amax / scales_max / val_max)
    s_dec_b = tl.max(tl.abs(x), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = (s_dec_b / s_dec).to(tl.float8e4nv).to(tl.float32)
    s_dec_b_e4m3 = tl.where(s_dec_b_e4m3 == 0, 1.0, s_dec_b_e4m3)
    return s_dec_b_e4m3, s_dec


@triton.jit
def _get_alt_scales(x, val_max, s_dec):
    s_dec_b = tl.max(tl.abs(x), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = (s_dec_b * (6 / 4) / s_dec).to(tl.float8e4nv).to(tl.float32)
    s_dec_b_e4m3 = tl.where(s_dec_b_e4m3 == 0, 1.0, s_dec_b_e4m3)
    return s_dec_b_e4m3


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def _rtn_1x16s_fp4_kernel(
    x_ptr, amax_ptr, output_ptr,
    n_elements: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    four_over_six: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))

    scales_max = 256.00 if four_over_six else 448.00
    val_max = 6.0 / scale_override
    amax = tl.load(amax_ptr)

    s_dec_b_e4m3, s_dec = _get_scales(x_grouped, amax, val_max, scales_max)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)

    x_fp4 = _rtn_fp4(x_scaled)
    x_dequantized = x_fp4 * (s_dec_b_e4m3 * s_dec)

    if not four_over_six:
        best_x_dequantized = x_dequantized
    else:
        alt_s_dec_b_e4m3 = _get_alt_scales(x_grouped, val_max, s_dec)
        alt_x_scaled = x_grouped / (alt_s_dec_b_e4m3 * s_dec)
        alt_x_fp4 = _rtn_fp4(alt_x_scaled)
        alt_x_dequantized = alt_x_fp4 * (alt_s_dec_b_e4m3 * s_dec)

        error_six = tl.sum((x_grouped - x_dequantized) * (x_grouped - x_dequantized), axis=-1, keep_dims=True)
        error_four = tl.sum((x_grouped - alt_x_dequantized) * (x_grouped - alt_x_dequantized), axis=-1, keep_dims=True)
        best_x_dequantized = tl.where(error_six <= error_four, x_dequantized, alt_x_dequantized)

    x_dequantized_flat = tl.reshape(best_x_dequantized, (BLOCK_SIZE,))
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def rtn_1x16s_fp4(x, scale_override: float, group_size: int, four_over_six: bool):
    x = x.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _rtn_1x16s_fp4_kernel[grid](
        x_ptr=x, amax_ptr=x.abs().max(), output_ptr=output,
        n_elements=n_elements, scale_override=scale_override,
        group_size=group_size, four_over_six=four_over_six,
    )
    return output



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def _eden_1x16s_fp4_kernel(
    x_ptr, hadamard_matrix_ptr, current_amax_ptr, output_ptr, next_amax_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(hadamard_dim, hadamard_dim)
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)

    tl.atomic_max(next_amax_ptr, tl.max(tl.abs(x_had)).to(tl.float32), sem="relaxed")

    x_grouped = tl.reshape(x_had, (BLOCK_SIZE // group_size, group_size))

    scales_max = 255.99
    val_max = 6.0 / scale_override
    amax = tl.load(current_amax_ptr)
    s_dec = tl.where(amax == 0.0, 1.0, amax / scales_max / val_max)

    s_dec_b = tl.max(tl.abs(x_grouped), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = (s_dec_b / s_dec).to(tl.float8e4nv).to(tl.float32)
    s_dec_b_e4m3 = tl.where(s_dec_b_e4m3 == 0, 1.0, s_dec_b_e4m3)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)

    x_scaled_abs = tl.abs(x_scaled)
    x_scaled_sign = tl.where(x_scaled > 0, 1, -1)
    x_fp4 = tl.where(
        x_scaled_abs >= 5, 6,
        tl.where(x_scaled_abs >= 3.5, 4,
        tl.where(x_scaled_abs >= 2.5, 3,
        tl.where(x_scaled_abs >= 1.75, 2,
        tl.where(x_scaled_abs >= 1.25, 1.5,
        tl.where(x_scaled_abs >= 0.75, 1,
        tl.where(x_scaled_abs >= 0.25, 0.5,
        0))))))) * x_scaled_sign

    x_scaled = tl.reshape(x_scaled, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_fp4 = tl.reshape(x_fp4, (BLOCK_SIZE // hadamard_dim, hadamard_dim))

    num = tl.sum(x_scaled * x_scaled, axis=-1, keep_dims=True)
    denom = tl.sum(x_scaled * x_fp4, axis=-1, keep_dims=True)
    correction = tl.where(denom == 0.0, 1.0, num / denom)

    scales = tl.reshape(s_dec_b_e4m3, (BLOCK_SIZE // hadamard_dim, hadamard_dim // group_size))
    corrected_scales = tl.reshape(scales * correction, (BLOCK_SIZE // group_size, 1))

    bitscales = tl.cast(corrected_scales.to(tl.float8e4nv), tl.uint8, bitcast=True)
    prevscale = tl.cast((bitscales - 1), tl.float8e4nv, bitcast=True).to(tl.float32)
    currscale = tl.cast((bitscales), tl.float8e4nv, bitcast=True).to(tl.float32)
    nextscale = tl.cast((bitscales + 1), tl.float8e4nv, bitcast=True).to(tl.float32)

    up = tl.where(currscale > corrected_scales, currscale, nextscale)
    down = tl.where(currscale > corrected_scales, prevscale, currscale)
    prob_up = (corrected_scales - down) / (up - down)

    scale_start_idx = pid * (BLOCK_SIZE // group_size)
    scale_offsets = scale_start_idx + tl.arange(0, BLOCK_SIZE // group_size)
    sampled_prob = tl.rand(seed, scale_offsets).reshape(BLOCK_SIZE // group_size, 1)

    scales = tl.where(sampled_prob < prob_up, up, down)
    scales = tl.reshape(scales, (BLOCK_SIZE // group_size, 1))
    x_fp4 = tl.reshape(x_fp4, (BLOCK_SIZE // group_size, group_size))

    x_dequantized = x_fp4 * scales * s_dec
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    tl.store(output_ptr + offsets, x_dequantized_flat.to(x_ptr.dtype.element_ty), mask=mask)


@torch.compiler.disable()
def eden_1x16s_fp4(x, hadamard_matrix, scale_override: float, group_size: int, current_amax):
    hadamard_dim = hadamard_matrix.size(0)
    x = x.contiguous()
    hadamard_matrix = hadamard_matrix.T.contiguous()
    output = torch.empty_like(x)
    seed = randint(0, 1_000_000)
    next_amax = torch.zeros_like(current_amax)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _eden_1x16s_fp4_kernel[grid](
        x_ptr=x, hadamard_matrix_ptr=hadamard_matrix,
        current_amax_ptr=current_amax, output_ptr=output,
        next_amax_ptr=next_amax, n_elements=n_elements,
        hadamard_dim=hadamard_dim, scale_override=scale_override,
        group_size=group_size, seed=seed,
    )
    return output, next_amax



class AmaxStorage:
    __slots__ = ("e_ht_amax", "weght_tht_amax", "e_tht_amax", "input_tht_amax")

    def __init__(self):
        self.e_ht_amax = None
        self.weght_tht_amax = None
        self.e_tht_amax = None
        self.input_tht_amax = None



class FakeQuartetFn(torch.autograd.Function):
    group_size = 16
    forward_scale_override = 1.0
    backward_scale_override = (17 / 16) * 0.93
    hadamard_matrix = None

    @torch.compile(dynamic=False)
    @staticmethod
    def forward(ctx, input, weight, amax_storage, delayed_amax, disable_forward_quant, disable_backward_quant, four_over_six):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        ctx.delayed_amax = delayed_amax
        ctx.amax_storage = amax_storage
        ctx.disable_backward_quant = disable_backward_quant

        if disable_forward_quant:
            input_fq = input
            weight_fq = weight
        else:
            input_fq = rtn_1x16s_fp4(input, FakeQuartetFn.forward_scale_override, FakeQuartetFn.group_size, four_over_six)
            weight_fq = rtn_1x16s_fp4(weight, FakeQuartetFn.forward_scale_override, FakeQuartetFn.group_size, four_over_six)

        ctx.save_for_backward(input_fq, weight_fq)
        return F.linear(input_fq, weight_fq)

    @staticmethod
    def backward(ctx, grad_output):
        input_fq, weight_fq = ctx.saved_tensors
        dtype = grad_output.dtype
        input_fq = input_fq.to(dtype).reshape(ctx.batch * ctx.seq, ctx.in_dim)
        weight_fq = weight_fq.to(dtype)
        grad_output = grad_output.reshape(ctx.batch * ctx.seq, ctx.out_dim)

        FakeQuartetFn.hadamard_matrix = rerotate_hadamard(FakeQuartetFn.hadamard_matrix)

        if ctx.disable_backward_quant:
            grad_input = F.linear(grad_output, weight_fq.T, None).view(ctx.batch, ctx.seq, ctx.in_dim)
            grad_weight = F.linear(grad_output.T, input_fq.T, None)
            return grad_input, grad_weight, None, None, None, None, None

        had = FakeQuartetFn.hadamard_matrix.to(grad_output.dtype)
        bso = FakeQuartetFn.backward_scale_override
        gs = FakeQuartetFn.group_size

        # EW: grad_output @ weight^T
        if ctx.amax_storage.e_ht_amax is None or not ctx.delayed_amax:
            ctx.amax_storage.e_ht_amax = (grad_output.reshape(-1, had.size(0)) @ had.T).abs().max().float()
        e_ht_fp4, ctx.amax_storage.e_ht_amax = eden_1x16s_fp4(grad_output, had, bso, gs, ctx.amax_storage.e_ht_amax)

        if ctx.amax_storage.weght_tht_amax is None or not ctx.delayed_amax:
            ctx.amax_storage.weght_tht_amax = (weight_fq.T.reshape(-1, had.size(0)) @ had.T).abs().max().float()
        weight_tht_fp4, ctx.amax_storage.weght_tht_amax = eden_1x16s_fp4(weight_fq.T, had, bso, gs, ctx.amax_storage.weght_tht_amax)

        grad_input = F.linear(e_ht_fp4, weight_tht_fp4, None).view(ctx.batch, ctx.seq, ctx.in_dim)

        # EtX: grad_output^T @ input
        if ctx.amax_storage.e_tht_amax is None or not ctx.delayed_amax:
            ctx.amax_storage.e_tht_amax = (grad_output.T.reshape(-1, had.size(0)) @ had.T).abs().max().float()
        e_tht_fp4, ctx.amax_storage.e_tht_amax = eden_1x16s_fp4(grad_output.T, had, bso, gs, ctx.amax_storage.e_tht_amax)

        if ctx.amax_storage.input_tht_amax is None or not ctx.delayed_amax:
            ctx.amax_storage.input_tht_amax = (input_fq.T.reshape(-1, had.size(0)) @ had.T).abs().max().float()
        input_tht_fp4, ctx.amax_storage.input_tht_amax = eden_1x16s_fp4(input_fq.T, had, bso, gs, ctx.amax_storage.input_tht_amax)

        grad_weight = F.linear(e_tht_fp4, input_tht_fp4, None)

        return grad_input, grad_weight, None, None, None, None, None



class FakeQuartetLinear(torch.nn.Linear):

    def __init__(self, *args, hadamard_dim=32, delayed_amax=False,
                 disable_forward_quant=False, disable_backward_quant=False,
                 four_over_six=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hadamard_dim = hadamard_dim
        self.delayed_amax = delayed_amax
        self.disable_forward_quant = disable_forward_quant
        self.disable_backward_quant = disable_backward_quant
        self.four_over_six = four_over_six
        self.amax_storage = AmaxStorage()

        if FakeQuartetFn.hadamard_matrix is None:
            FakeQuartetFn.hadamard_matrix = get_hadamard_matrix(
                self.hadamard_dim, dtype=torch.float32, device="cuda",
            )

    def forward(self, x):
        return FakeQuartetFn.apply(
            x, self.weight, self.amax_storage,
            self.delayed_amax, self.disable_forward_quant,
            self.disable_backward_quant, self.four_over_six,
        )
