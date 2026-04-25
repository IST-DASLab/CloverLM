import pytest
import torch

pytest.importorskip("qutlass")
from quartet2.linear import (
    Quartet_II_linear,
    _dq_gridflip_weight_fp4,
    _fp4_mm,
    _fp4_mm_gridflip,
    _quant_gridflip_weight_fp4,
    _quant_gridflip_weight_fp4_reference,
    fp4_mm_backend,
    get_fp4_mm_backend,
    get_gridflip_shift,
    get_fp4_weight_quantizer,
    set_fp4_weight_quantizer,
)
from quartet2.quant import NVFP4QuantMode, quant_fp4


if not torch.cuda.is_available():
    pytest.skip("CUDA required for these tests.", allow_module_level=True)


def _quantize_pair(m: int, n: int, k: int):
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    xq = quant_fp4(
        x,
        amax=x.abs().max().float(),
        scale_override=1.0,
        mode=NVFP4QuantMode.FOUR_SIX,
    )
    wq = quant_fp4(
        w,
        amax=w.abs().max().float(),
        scale_override=1.0,
        mode=NVFP4QuantMode.FOUR_SIX,
    )
    return xq, wq


@pytest.mark.parametrize("shape", [(128, 256, 128), (512, 256, 128), (128, 384, 256)])
@torch.inference_mode()
def test_qutlass_backend_matches_flashinfer_and_dequantized(shape):
    torch.manual_seed(0)
    m, n, k = shape
    xq, wq = _quantize_pair(m, n, k)
    alpha = xq.tensor_scale * wq.tensor_scale

    old_backend = get_fp4_mm_backend()
    with fp4_mm_backend("flashinfer"):
        flashinfer_out = _fp4_mm(
            xq.fp4,
            wq.fp4,
            xq.micro_scales,
            wq.micro_scales,
            alpha,
        )
    with fp4_mm_backend("qutlass"):
        qutlass_out = _fp4_mm(
            xq.fp4,
            wq.fp4,
            xq.micro_scales,
            wq.micro_scales,
            alpha,
        )
    with fp4_mm_backend("dequantized"):
        dequantized_out = _fp4_mm(
            xq.fp4,
            wq.fp4,
            xq.micro_scales,
            wq.micro_scales,
            alpha,
        )

    assert get_fp4_mm_backend() == old_backend
    assert qutlass_out.equal(flashinfer_out)
    torch.testing.assert_close(qutlass_out, dequantized_out, rtol=2e-2, atol=2.0)


@torch.inference_mode()
def test_qutlass_backend_matches_flashinfer_linear_forward():
    torch.manual_seed(1)
    linear = Quartet_II_linear(128, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn((1, 512, 128), device="cuda", dtype=torch.bfloat16)

    with fp4_mm_backend("flashinfer"):
        flashinfer_out = linear(x)
    with fp4_mm_backend("qutlass"):
        qutlass_out = linear(x)
    with fp4_mm_backend("dequantized"):
        dequantized_out = linear(x)

    assert qutlass_out.equal(flashinfer_out)
    torch.testing.assert_close(qutlass_out, dequantized_out, rtol=2e-2, atol=2.0)


@torch.inference_mode()
def test_qutlass_backend_compiles_forward():
    torch.manual_seed(2)
    linear = Quartet_II_linear(128, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn((1, 128, 128), device="cuda", dtype=torch.bfloat16)

    def fwd(inp):
        return linear(inp)

    with fp4_mm_backend("qutlass"):
        out = torch.compile(fwd, fullgraph=True)(x)

    assert out.shape == (1, 128, 256)


def test_qutlass_backend_matches_flashinfer_backward():
    torch.manual_seed(3)
    weight = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    x_ref = torch.randn((1, 128, 128), device="cuda", dtype=torch.bfloat16)
    grad = torch.randn((1, 128, 256), device="cuda", dtype=torch.bfloat16)

    def run(backend: str, backward_backend: str | None = None):
        linear = Quartet_II_linear(128, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(weight)
        x = x_ref.clone().requires_grad_(True)
        torch.manual_seed(100)
        with fp4_mm_backend(backend):
            y = linear(x)
        torch.manual_seed(200)
        if backward_backend is None:
            y.backward(grad)
        else:
            with fp4_mm_backend(backward_backend):
                y.backward(grad)
        return y.detach(), x.grad.detach(), linear.weight.grad.detach()

    flashinfer_out, flashinfer_x_grad, flashinfer_w_grad = run("flashinfer")
    qutlass_out, qutlass_x_grad, qutlass_w_grad = run("qutlass")
    pinned_out, pinned_x_grad, pinned_w_grad = run("qutlass", backward_backend="dequantized")

    assert qutlass_out.equal(flashinfer_out)
    assert qutlass_x_grad.equal(flashinfer_x_grad)
    assert qutlass_w_grad.equal(flashinfer_w_grad)
    assert pinned_out.equal(qutlass_out)
    assert pinned_x_grad.equal(qutlass_x_grad)
    assert pinned_w_grad.equal(qutlass_w_grad)


@pytest.mark.parametrize("scale_override", [1.0, 0.875])
@torch.inference_mode()
def test_gridflip_quantizer_does_not_exceed_python_reference_error(scale_override):
    torch.manual_seed(4)
    n, k = 256, 128
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    wq = _quant_gridflip_weight_fp4(
        w,
        amax=w.abs().max().float(),
        scale_override=scale_override,
        grid_shift=0.25,
        mode=NVFP4QuantMode.FOUR_SIX,
    )
    ref_wq = _quant_gridflip_weight_fp4_reference(
        w,
        amax=w.abs().max().float(),
        scale_override=scale_override,
        grid_shift=0.25,
        mode=NVFP4QuantMode.FOUR_SIX,
    )
    flags = (wq.micro_scales.view(torch.uint8) & 0x80) != 0
    assert flags.any()
    assert wq.tensor_scale.equal(ref_wq.tensor_scale)
    cuda_dq = _dq_gridflip_weight_fp4(wq.fp4, wq.micro_scales, 0.25).float() * wq.tensor_scale.float()
    ref_dq = _dq_gridflip_weight_fp4(ref_wq.fp4, ref_wq.micro_scales, 0.25).float() * ref_wq.tensor_scale.float()
    cuda_error = (cuda_dq - w.float()).square().reshape(n, k // 16, 16).sum(dim=-1)
    ref_error = (ref_dq - w.float()).square().reshape(n, k // 16, 16).sum(dim=-1)
    assert torch.all(cuda_error <= ref_error + 1e-5)


@torch.inference_mode()
def test_gridflip_matmul_matches_dequantized_reference():
    torch.manual_seed(4)
    m, n, k = 128, 256, 128
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    xq = quant_fp4(
        x,
        amax=x.abs().max().float(),
        scale_override=1.0,
        mode=NVFP4QuantMode.FOUR_SIX,
    )
    wq = _quant_gridflip_weight_fp4(
        w,
        amax=w.abs().max().float(),
        scale_override=1.0,
        grid_shift=0.25,
        mode=NVFP4QuantMode.FOUR_SIX,
    )
    alpha = xq.tensor_scale * wq.tensor_scale

    with fp4_mm_backend("qutlass"):
        qutlass_out = _fp4_mm_gridflip(
            xq.fp4,
            wq.fp4,
            xq.micro_scales,
            wq.micro_scales,
            alpha,
            grid_shift=0.25,
        )
    with fp4_mm_backend("dequantized"):
        dequantized_out = _fp4_mm_gridflip(
            xq.fp4,
            wq.fp4,
            xq.micro_scales,
            wq.micro_scales,
            alpha,
            grid_shift=0.25,
        )

    torch.testing.assert_close(qutlass_out, dequantized_out, rtol=2e-2, atol=2.0)


def test_gridflip_linear_forward_matches_dequantized_and_pins_backward_backend():
    torch.manual_seed(5)
    weight = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    x_ref = torch.randn((1, 128, 128), device="cuda", dtype=torch.bfloat16)
    grad = torch.randn((1, 128, 256), device="cuda", dtype=torch.bfloat16)

    def run(backend: str, backward_backend: str | None = None):
        old_quantizer = get_fp4_weight_quantizer()
        old_shift = get_gridflip_shift()
        set_fp4_weight_quantizer("gridflip", gridflip_shift=0.25)
        linear = Quartet_II_linear(128, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(weight)
        x = x_ref.clone().requires_grad_(True)
        try:
            with fp4_mm_backend(backend):
                y = linear(x)
            torch.manual_seed(200)
            if backward_backend is None:
                y.backward(grad)
            else:
                with fp4_mm_backend(backward_backend):
                    y.backward(grad)
        finally:
            set_fp4_weight_quantizer(old_quantizer, gridflip_shift=old_shift)
        return y.detach(), x.grad.detach(), linear.weight.grad.detach()

    qutlass_out, qutlass_x_grad, qutlass_w_grad = run("qutlass")
    pinned_out, pinned_x_grad, pinned_w_grad = run("qutlass", backward_backend="dequantized")
    dequantized_out, _, _ = run("dequantized")

    torch.testing.assert_close(qutlass_out, dequantized_out, rtol=2e-2, atol=2.0)
    assert pinned_out.equal(qutlass_out)
    assert pinned_x_grad.equal(qutlass_x_grad)
    assert pinned_w_grad.equal(qutlass_w_grad)
