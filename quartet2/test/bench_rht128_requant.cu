#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <string>
#include <cstdlib>
#include <iostream>
#include <random>


// rht128_requant launcher from csrc/rht128_requant.cu
void rht128_requant(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    long seed, float fp4_max, float fp8_max, int M, int N);

static inline nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Initialize identity matrix (128x128) in BF16
void init_identity_matrix_128(nv_bfloat16* trans) {
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            trans[i * 128 + j] = float_to_bfloat16(i == j ? 1.0f : 0.0f);
        }
    }
}

// Fill packed FP4x2 buffer with random nibbles (host-side)
void init_random_fp4x2(__nv_fp4x2_storage_t* buf, size_t elems, std::mt19937& gen) {
    std::uniform_int_distribution<int> byte_dist(0, 255);
    unsigned char* raw = reinterpret_cast<unsigned char*>(buf);
    for (size_t i = 0; i < elems; ++i) {
        raw[i] = static_cast<unsigned char>(byte_dist(gen));
    }
}

// Fill FP8 E4M3 scales with a constant value (e.g., 1.0f)
void init_fp8_scales(__nv_fp8_e4m3* s, size_t elems, float value) {
    // FP8 E4M3 is supported by CUDA types; reinterpret as float and convert via __nv_fp8_e4m3 constructor
    for (size_t i = 0; i < elems; ++i) {
        // Use CUDA intrinsic for rounding if available; otherwise rely on implicit conversion
        s[i] = __nv_fp8_e4m3(value);
    }
}

int main(int argc, char** argv) {
    std::mt19937 gen(42);

    bool transpose = false; // kept for parity with other benches, not used here
    if (argc > 1) transpose = atoi(argv[1]) == 1;

    // Problem sizes (must be divisible by 128)
    int M = 16384;
    int N = 4096;

    if (M % 128 != 0 || N % 128 != 0) {
        std::cerr << "M and N must be divisible by 128" << std::endl;
        return 1;
    }

    // Host buffers
    nv_bfloat16* h_trans = new nv_bfloat16[128 * 128];
    init_identity_matrix_128(h_trans);

    // Input X is already quantized FP4 (packed as fp4x2)
    size_t x_q_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) / 2; // two FP4 per byte
    __nv_fp4x2_storage_t* h_x_q = reinterpret_cast<__nv_fp4x2_storage_t*>(new unsigned char[x_q_bytes]);
    init_random_fp4x2(h_x_q, x_q_bytes, gen);

    size_t x_scales_elems = static_cast<size_t>(M) * static_cast<size_t>(N) / 16;
    __nv_fp8_e4m3* h_x_scales = new __nv_fp8_e4m3[x_scales_elems];
    init_fp8_scales(h_x_scales, x_scales_elems, 1.0f);

    float h_x_global_scale = 1.0f; // multiply with FP8 group scales inside kernel

    // Device buffers
    nv_bfloat16 *d_trans = nullptr, *d_scratch = nullptr;
    __nv_fp4x2_storage_t *d_y = nullptr, *d_x_q = nullptr;
    __nv_fp8_e4m3 *d_scales_fp8 = nullptr, *d_x_scales = nullptr;
    unsigned* d_max_scale = nullptr;
    float *d_global_scale = nullptr, *d_x_global_scale = nullptr;

    cudaMalloc(&d_trans, 128 * 128 * sizeof(nv_bfloat16));
    cudaMalloc(&d_x_q, x_q_bytes);
    cudaMalloc(&d_x_scales, x_scales_elems * sizeof(__nv_fp8_e4m3));

    // Output buffers (like bench_group_transform_eden)
    cudaMalloc(&d_scratch, static_cast<size_t>(M) * N * sizeof(nv_bfloat16) / 16);
    cudaMalloc(&d_scales_fp8, static_cast<size_t>(M) * N * sizeof(__nv_fp8_e4m3) / 16);
    cudaMalloc(&d_y, static_cast<size_t>(M) * N * sizeof(__nv_fp4x2_storage_t) / 2);
    cudaMalloc(&d_max_scale, sizeof(unsigned));
    cudaMalloc(&d_global_scale, sizeof(float));
    cudaMalloc(&d_x_global_scale, sizeof(float));

    cudaMemset(d_y, 0, static_cast<size_t>(M) * N * sizeof(__nv_fp4x2_storage_t) / 2);

    // Copy host -> device
    cudaMemcpy(d_trans, h_trans, 128 * 128 * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_q, h_x_q, x_q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_scales, h_x_scales, x_scales_elems * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_global_scale, &h_x_global_scale, sizeof(float), cudaMemcpyHostToDevice);

    // Run the requantization benchmark kernel
    // fp4_max/fp8_max are clipping ranges for output rounding; keep consistent with other benches
    float fp4_max = 6.0f;
    float fp8_max = 256.0f;
    long seed = 42;

    rht128_requant(
        d_y, d_scales_fp8, d_global_scale,
        d_scratch, d_max_scale, d_trans,
        d_x_q, d_x_scales, d_x_global_scale,
        seed, fp4_max, fp8_max, M, N);

    cudaDeviceSynchronize();

    // Cleanup
    delete[] h_trans;
    delete[] h_x_scales;
    delete[] reinterpret_cast<unsigned char*>(h_x_q);
    cudaFree(d_trans);
    cudaFree(d_x_q);
    cudaFree(d_x_scales);
    cudaFree(d_scratch);
    cudaFree(d_scales_fp8);
    cudaFree(d_y);
    cudaFree(d_max_scale);
    cudaFree(d_global_scale);
    cudaFree(d_x_global_scale);

    std::cout << "Done." << std::endl;
    return 0;
}
