#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

// Kernel declaration (to be implemented)
void eden_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long seed, long rows, long cols);
void four_six_fp4(__nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr, float* global_scale_ptr, const nv_bfloat16* x_ptr, const float* amax_ptr, float scale_override, long rows, long cols);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void initialize_input(nv_bfloat16* h_x, float* h_amax, long nelem) {
    const int G = 128;
    std::mt19937 rng(42);
    std::normal_distribution<float> stddev_dist(0.0f, 2.0f);
    *h_amax = 0.0f;
    
    for (long i = 0; i < nelem; i += G) {
        float stddev = std::abs(stddev_dist(rng));
        std::normal_distribution<float> value_dist(0.0f, stddev);
        
        for (int j = 0; j < G && (i + j) < nelem; j++) {
            float val = value_dist(rng);
            h_x[i + j] = __float2bfloat16(val);
            *h_amax = std::max(*h_amax, std::abs(val));
        }
    }
    
    std::cout << "Initialized " << nelem << " elements, amax = " << *h_amax << std::endl;
}

void initialize_output(uint8_t* h_y, __nv_fp8_e4m3* h_scale, long nelem) {
    const int G = 128;
    long num_y = nelem / 2;
    long num_scales = nelem / G;
    
    // Zero out FP4 output
    memset(h_y, 0, num_y);
    
    // Initialize scales to -1 (in FP8 E4M3 format)
    for (long i = 0; i < num_scales; i++) {
        h_scale[i] = __nv_fp8_e4m3(-1.0f);  // Convert -1 to E4M3
    }
    
    std::cout << "Initialized " << num_y << " output bytes and " 
              << num_scales << " scales" << std::endl;
}

static __host__ __device__ __forceinline__ float2 cvt_fp4x2_to_float2(__nv_fp4x2_storage_t x) {
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(x, __nv_fp4_interpretation_t::__NV_E2M1);
    __half_raw r1, r2;
    r1.x = raw.x;
    r2.x = raw.y;
    return make_float2(__half2float(__half(r1)), __half2float(__half(r2)));
}

void dequantize_fp4(const uint8_t* h_y, const __nv_fp8_e4m3* h_scale,
                    float* h_dequant, long nelem, float global_abs_max) {
    // FP4 E2M1 value range
    const float val_max = 6.0f;  // Maximum representable value in FP4 E2M1

    // E4M3 scale range
    //const float scales_max = 448.0f;  // Maximum representable value in FP8 E4M3
    const float scales_max = 255.99f;  // Maximum representable value in FP8 E4M3

    // Calculate global scale factor
    float global_scale = (global_abs_max == 0.0f) ? 1.0f : global_abs_max / scales_max / val_max;

    std::cout << "Dequantizing with global_scale = " << global_scale << std::endl;

    // Process each element
    for (long i = 0; i < nelem; i+=2) {
        float2 as_float = cvt_fp4x2_to_float2(h_y[i/2]);
        // Get scale for this group of 16 elements
        long scale_idx = i / 16;
        float scale_val = float(h_scale[scale_idx]);
        if (!isfinite(scale_val) && i % 16 == 0) {
            std::cout << std::hex << (unsigned)reinterpret_cast<const std::uint8_t&>(h_scale[scale_idx]) << std::dec << std::endl;
        }

        // Dequantize: scale * global_scale * fp4_value
        h_dequant[i] = scale_val * global_scale * as_float.x;
        h_dequant[i+1] = scale_val * global_scale * as_float.y;
    }

    std::cout << "Dequantization complete" << std::endl;
}

struct QuantizationError {
    float mean_error;
    float max_abs_error;
    float mean_abs_error;
    float mean_squared_error;
};

QuantizationError calculate_quantization_error(const float* h_dequant, const nv_bfloat16* h_ref, long nelem) {
    double mean_error = 0.0;
    float max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double mean_squared_error = 0.0;
    for (int i = 0; i < nelem; i++) {
        float g = h_dequant[i];
        float r = __bfloat162float(h_ref[i]);
        mean_error += g - r;
        max_abs_error = std::max(max_abs_error, std::fabs(g - r));
        mean_abs_error += std::fabs(g - r);
        mean_squared_error += (g - r) * (g - r);
    }
    mean_error /= nelem;
    mean_abs_error /= nelem;
    mean_squared_error /= nelem;
    return QuantizationError{float(mean_error), float(max_abs_error), float(mean_abs_error), float(mean_squared_error)};
}

enum ALGO {
    EDEN,
    FOUR_SIX
};

void run_test(ALGO algo) {
    const int N = 8192;
    const int M = 2048;
    const int G = 16;
    const int H = 128;
    const long seed = 42;
    const long nelem = static_cast<long>(N) * M;
    
    // Allocate host memory
    nv_bfloat16* h_x = new nv_bfloat16[nelem];
    uint8_t* h_y = new uint8_t[nelem / 2];
    __nv_fp8_e4m3* h_scale = new __nv_fp8_e4m3[nelem / G];
    float h_amax;
    
    // Initialize data
    initialize_input(h_x, &h_amax, nelem);
    initialize_output(h_y, h_scale, nelem);
    
    // Allocate device memory
    nv_bfloat16* d_x;
    __nv_fp4x4_e2m1* d_y;
    __nv_fp8_e4m3* d_scales;
    float* d_amax;
    float* d_global_scale;

    CUDA_CHECK(cudaMalloc(&d_x, nelem * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_y, (nelem / 2) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_scales, (nelem / G) * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_amax, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global_scale, sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, nelem * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, nelem / 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scale, (nelem / G) * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_amax, &h_amax, sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    float scale_override = 1.0f;

    std::cout << "Launching kernel" << std::endl;

    if (algo == EDEN) {
        eden_fp4(d_y, d_scales, d_global_scale, d_x, d_amax, scale_override, seed, M, N);
    } else if (algo == FOUR_SIX) {
        four_six_fp4(d_y, d_scales, d_global_scale, d_x, d_amax, scale_override, M,  N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_y, d_y, nelem / 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scale, d_scales, (nelem / G) * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));
    
    std::cout << "Kernel completed successfully" << std::endl;
    
    // Print first few results for verification
    std::cout << "First 16 input values (BF16): ";
    for (int i = 0; i < 16; i++) {
        std::cout << __bfloat162float(h_x[i]) << " ";
    }
    std::cout << std::endl;
    
    std::cout << "First 8 output bytes (FP4 packed): ";
    for (int i = 0; i < 8; i++) {
        std::cout << std::hex << (int)h_y[i] << " ";
    }
    std::cout << std::dec << std::endl;
    std::cout << "With scale: " << (float)h_scale[0] << std::endl;

    // OK; now check round trip
    float* h_dequant = new float[nelem];
    dequantize_fp4(h_y, h_scale, h_dequant, nelem, h_amax);

    std::cout << "First 16 output values (dequant): ";
    for (int i = 0; i < 16; i++) {
        std::cout << h_dequant[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    auto errors = calculate_quantization_error(h_dequant, h_x, nelem);
    std::cout << "Mean error: " << errors.mean_error << std::endl;
    std::cout << "Max abs error: " << errors.max_abs_error << std::endl;
    std::cout << "Mean abs error: " << errors.mean_abs_error << std::endl;
    std::cout << "Mean squared error: " << errors.mean_squared_error << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_amax));
    
    delete[] h_x;
    delete[] h_y;
    delete[] h_scale;
    delete[] h_dequant;
}

int main() {
    run_test(EDEN);
    printf("\n\n");
    run_test(FOUR_SIX);
    return 0;
}