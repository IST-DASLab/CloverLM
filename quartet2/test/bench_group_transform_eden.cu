#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>
#include <cuda_fp4.h>
#include <random>

void group_transform_128_eden(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N, bool transposeX);

void transform_rht128_eden(__nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N, bool transpose);

// Helper function to convert float to bfloat16
__host__ __device__ nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert bfloat16 to float
__host__ __device__ float bfloat16_to_float(nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// Initialize identity matrix (128x128)
void init_identity_matrix(nv_bfloat16* trans) {
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            float val = (i == j) ? 1.0f : 0.0f;
            trans[i * 128 + j] = float_to_bfloat16(val);
        }
    }
}

// Initialize matrix with random values
void init_random_matrix(nv_bfloat16* matrix, int rows, int cols, std::mt19937& gen) {
    std::uniform_int_distribution<int> dis(0, 9);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = float_to_bfloat16(dis(gen));
    }
}

int main(int argc, char** argv) {
    std::mt19937 gen(42); // Fixed seed for reproducibility

    bool transpose = false;
    bool had128 = false;
    if (argc > 1) {
        transpose = atoi(argv[1]) == 1;
    }
    if (argc > 2) {
        had128 = std::string(argv[2]) == "had";
    }

    int M = 16384, N = 4096;

    // Allocate host memory
    nv_bfloat16* h_trans = new nv_bfloat16[128 * 128];
    nv_bfloat16* h_x = new nv_bfloat16[M * N];

    // Initialize trans as identity
    init_identity_matrix(h_trans);

    // Initialize x with random values
    init_random_matrix(h_x, M, N, gen);

    nv_bfloat16 *d_trans, *d_x, *d_s;
    __nv_fp4x2_storage_t* d_y;
    __nv_fp8_e4m3* d_scales_fp8;
    unsigned* d_max_scale;
    float* d_global_scale;

    // Allocate device memory
    cudaMalloc(&d_trans, 128 * 128 * sizeof(nv_bfloat16));
    cudaMalloc(&d_x, M * N * sizeof(nv_bfloat16));
    cudaMalloc(&d_s, M * N * sizeof(nv_bfloat16) / 16);
    cudaMalloc(&d_scales_fp8, M * N * sizeof(__nv_fp8_e4m3) / 16);
    cudaMalloc(&d_y, M * N * sizeof(__nv_fp4x2_storage_t) / 2);
    cudaMalloc(&d_max_scale, sizeof(unsigned));
    cudaMalloc(&d_global_scale, sizeof(float));
    cudaMemset(d_y, 0, M * N * sizeof(__nv_fp4x2_storage_t) / 2);

    // Copy to device
    cudaMemcpy(d_trans, h_trans, 128 * 128 * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, M * N * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);

    // Run kernel
    if (had128) {
        transform_rht128_eden(d_y, d_scales_fp8, d_global_scale, d_s, d_max_scale, d_trans, d_x, 42, 6, 256, M, N,  transpose);
    } else {
        group_transform_128_eden(d_y, d_scales_fp8, d_global_scale, d_s, d_max_scale, d_trans, d_x, 42, 6, 256, M, N,  transpose);
    }

    cudaDeviceSynchronize();
    // Cleanup
    delete[] h_trans;
    delete[] h_x;
    cudaFree(d_trans);
    cudaFree(d_x);
    cudaFree(d_y);
    std::cout << std::endl;
    return 0;
}
