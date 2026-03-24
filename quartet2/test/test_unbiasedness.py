import torch
from scipy.linalg import hadamard
import quartet2.quant
from quartet2.linear import unblock


torch.random.manual_seed(42)


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
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
            dtype=hadamard_matrix.dtype
        ) * 2 - 1
    )
    return hadamard_matrix @ signs # NOTE: rerotate along last dim, inner dim for TN GEMM


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
        dtype=torch.float64,
        device=device,
    )
    x_fp4_dq = grid_dq[x_e2m1_unpacked]
    scales_dq = x_e4m3.to(torch.float32)
    scales_dq = unblock(scales_dq, x_e2m1.shape[0], x_e2m1.shape[1] * 2)
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq[..., None]).flatten(
        start_dim=-2
    ) * alpha
    return x_dq


M = 1024
N = 1024
K = 1024

def test_unbiasedness(transposeA, transposeB, matmul_in_fp4, scale_override=1.0):
    HADAMARD_DIM = 128

    A = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    B = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
    ht = get_hadamard_matrix(HADAMARD_DIM, A.dtype, A.device).to(torch.bfloat16)



    with torch.no_grad():
        for acc_steps in [1, 4, 16, 64, 256, 1024]:
            accumulator = torch.zeros_like(A @ B.T, dtype=torch.float32)
            for i in range(acc_steps):
                ht = rerotate_hadamard(ht)

                rng1 = torch.randint(0, 2**32, ())
                rng2 = torch.randint(0, 2**32, ())

                A_q, A_group_scales, A_tensor_scale = quartet2.quant.quant_had_eden(
                    h=ht, x=A, seed=rng1, transpose=transposeA, scale_override=scale_override
                )
                B_q, B_group_scales, B_tensor_scale = quartet2.quant.quant_had_eden(
                    h=ht, x=B, seed=rng2, transpose=transposeB, scale_override=scale_override
                )

                if matmul_in_fp4:
                    import qutlass
                    accumulator += qutlass.matmul_nvf4_bf16_tn(A_q, B_q, A_group_scales, B_group_scales, alpha=A_tensor_scale*B_tensor_scale)
                else:
                    accumulator += _dq_fp4(A_q, A_group_scales, A_tensor_scale.item()) @ _dq_fp4(B_q, B_group_scales, B_tensor_scale.item()).T
            accumulator /= acc_steps

            if transposeA and not transposeB:
                quad_err = (accumulator - A.T @ B.T).pow(2).mean() / (A.T @ B.T).pow(2).mean()
            elif transposeA and transposeB:
                quad_err = (accumulator - A.T @ B).pow(2).mean() / (A.T @ B).pow(2).mean()
            elif transposeB:
                quad_err = (accumulator - A @ B).pow(2).mean() / (A @ B).pow(2).mean()
            else:
                quad_err = (accumulator - A @ B.T).pow(2).mean() / (A @ B.T).pow(2).mean()
            eff_bitwidth = (-torch.log2(quad_err) / 2).item()
            print(f"{acc_steps:>4}: {eff_bitwidth:.2f} bits")

for so in [1.0, (17 / 16) * 0.93]:
    for matmul_in_fp4 in [False, True]:
        print(f"t=false, false, qutlass={matmul_in_fp4}, so={so:.2f}")
        test_unbiasedness(False, False, matmul_in_fp4, scale_override=so)
        print(f"t=true, false, qutlass={matmul_in_fp4}, so={so:.2f}")
        test_unbiasedness(True, False, matmul_in_fp4, scale_override=so)
        print(f"t=false, true, qutlass={matmul_in_fp4}, so={so:.2f}")
        test_unbiasedness(False, True, matmul_in_fp4, scale_override=so)
        print(f"t=true, true, qutlass={matmul_in_fp4}, so={so:.2f}")
        test_unbiasedness(True, True, matmul_in_fp4, scale_override=so)
    # NEED TO GROW BY ~1 bit per 4x samples
