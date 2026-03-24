#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>
#include <random>

void group_transform_128(nv_bfloat16* y, const nv_bfloat16* trans, const nv_bfloat16* x, int M, int N, bool transpose);
void transform_rht128(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);
void transform_rht128_ws(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);
void transform_rht128_tma(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);

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

// Initialize matrix with constant value
void init_constant_matrix(nv_bfloat16* matrix, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = float_to_bfloat16(val);
    }
}

// Check if two matrices are approximately equal
bool matrices_equal(const nv_bfloat16* a, const nv_bfloat16* b, int size, float tolerance = 1e-2) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(bfloat16_to_float(a[i]) - bfloat16_to_float(b[i]));
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": "
                      << bfloat16_to_float(a[i]) << " vs "
                      << bfloat16_to_float(b[i]) << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// Print matrix for debugging
void print_matrix(const nv_bfloat16* matrix, int rows, int cols, const char* name) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    int max_print_rows = std::min(rows, 32);
    int max_print_cols = std::min(cols, 32);
    for (int i = 0; i < max_print_rows; i++) {
        for (int j = 0; j < max_print_cols; j++) {
            std::cout << bfloat16_to_float(matrix[i * cols + j]) << " ";
            if ((j + 1) % 16 == 0)
                std::cout << " ";
        }
        if (cols > max_print_cols) std::cout << "...";
        if ((i+1) % 16 == 0)
            std::cout << std::endl;
        std::cout << std::endl;
    }
    if (rows > max_print_rows) std::cout << "..." << std::endl;
    std::cout << std::endl;
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

    // Test 1: Identity transform (trans = I, should have y = x)
    {
        std::cout << "=== Test 1: Identity Transform ===" << std::endl;
        int M = 16384, N = 4096;

        nv_bfloat16 *h_trans, *h_x, *h_y, *h_expected;
        nv_bfloat16 *d_trans, *d_x, *d_y;

        // Allocate host memory
        h_trans = new nv_bfloat16[128 * 128];
        h_x = new nv_bfloat16[M * N];
        h_y = new nv_bfloat16[M * N];
        h_expected = new nv_bfloat16[M * N];

        // Initialize trans as identity
        init_identity_matrix(h_trans);

        // Initialize x with random values
        init_random_matrix(h_x, M, N, gen);

        // Expected output is same as input for identity
        memcpy(h_expected, h_x, M * N * sizeof(nv_bfloat16));

        // Allocate device memory
        cudaMalloc(&d_trans, 128 * 128 * sizeof(nv_bfloat16));
        cudaMalloc(&d_x, M * N * sizeof(nv_bfloat16));
        cudaMalloc(&d_y, M * N * sizeof(nv_bfloat16));
        cudaMemset(d_y, 0, M * N * sizeof(nv_bfloat16));

        // Copy to device
        cudaMemcpy(d_trans, h_trans, 128 * 128 * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, M * N * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);

        // Run kernel
        if (had128) {
            transform_rht128_tma(d_y, d_trans, d_x, M, N, transpose);
        } else {
            group_transform_128(d_y, d_trans, d_x, M, N, transpose);
        }
        cudaDeviceSynchronize();

        // Copy result back
        cudaMemcpy(h_y, d_y, M * N * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

        // Verify
        if (matrices_equal(h_y, h_expected, M * N)) {
            std::cout << "✓ Test 1 PASSED" << std::endl;
        } else {
            std::cout << "✗ Test 1 FAILED" << std::endl;
            print_matrix(h_x, M, N, "Input X");
            print_matrix(h_y, M, N, "Output Y");
        }

        // Cleanup
        delete[] h_trans;
        delete[] h_x;
        delete[] h_y;
        delete[] h_expected;
        cudaFree(d_trans);
        cudaFree(d_x);
        cudaFree(d_y);
        std::cout << std::endl;
    }

    std::cout << "All tests completed!" << std::endl;
    return 0;
}
