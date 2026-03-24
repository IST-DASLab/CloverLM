#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>
#include <random>

void group_transform_128(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose);

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

bool transpose_matrix(nv_bfloat16* trans, int rows, int cols) {
    std::vector<nv_bfloat16> tmp(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            tmp[j * rows + i] = trans[i * cols + j];
        }
    }

    for (int i = 0; i < rows * cols; i++) {
        trans[i] = tmp[i];
    }
    return true;
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
    int max_print_rows = std::min(rows, 4);
    int max_print_cols = std::min(cols, 256);
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

void test_identity_transform(bool transpose) {
    std::mt19937 gen(42);
    constexpr int H = 128;
    constexpr int M = 128, N = 256;

    nv_bfloat16 *d_trans, *d_x, *d_y;

    // Allocate host memory
    std::vector<nv_bfloat16> h_h(H * H);
    std::vector<nv_bfloat16> h_x(M * N);
    std::vector<nv_bfloat16> h_y(M * N);
    std::vector<nv_bfloat16> h_e(M * N);

    // Initialize trans as identity
    init_identity_matrix(h_h.data());

    // Initialize x with random values
    init_random_matrix(h_x.data(), M, N, gen);

    // Expected output is same as input for identity
    h_e = h_x;

    // Allocate device memory
    cudaMalloc(&d_trans, H * H * sizeof(nv_bfloat16));
    cudaMalloc(&d_x, M * N * sizeof(nv_bfloat16));
    cudaMalloc(&d_y, M * N * sizeof(nv_bfloat16));
    cudaMemset(d_y, 0, M * N * sizeof(nv_bfloat16));

    // Copy to device
    cudaMemcpy(d_trans, h_h.data(), H * H * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), M * N * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);

    // Run kernel
    group_transform_128(d_y, d_trans, d_x, M, N, transpose);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_y.data(), d_y, M * N * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

    if (transpose)
        transpose_matrix(h_e.data(), M, N);

    // Verify
    if (matrices_equal(h_y.data(), h_e.data(), M * N)) {
        std::cout << "✓ Test 1 PASSED" << std::endl;
    } else {
        std::cout << "✗ Test 1 FAILED" << std::endl;
        print_matrix(h_x.data(), M, N, "Input");
        if (transpose) {
            print_matrix(h_e.data(), N, M, "Expected");
            print_matrix(h_y.data(), N, M, "Output");
        } else {
            print_matrix(h_y.data(), M, N, "Output");
        }
    }

    // Cleanup
    cudaFree(d_trans);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    std::mt19937 gen(42); // Fixed seed for reproducibility

    // Test 1: Identity transform (trans = I, should have y = x)
    {
        std::cout << "=== Test 1: Identity Transform NT ===" << std::endl;
        test_identity_transform(false);
        std::cout << std::endl;

        std::cout << "=== Test 1: Identity Transform TT ===" << std::endl;
        test_identity_transform(true);
        std::cout << std::endl;
    }

    // Test 2: Zero transform (trans = 0, should have y = 0)
    {
        std::cout << "=== Test 2: Zero Transform ===" << std::endl;
        int M = 128, N = 256;

        nv_bfloat16 *h_trans, *h_x, *h_y, *h_expected;
        nv_bfloat16 *d_trans, *d_x, *d_y;

        h_trans = new nv_bfloat16[128 * 128];
        h_x = new nv_bfloat16[M * N];
        h_y = new nv_bfloat16[M * N];
        h_expected = new nv_bfloat16[M * N];

        // Initialize trans as zero matrix
        init_constant_matrix(h_trans, 128, 128, 0.0f);
        init_random_matrix(h_x, M, N, gen);
        init_constant_matrix(h_expected, M, N, 0.0f);

        cudaMalloc(&d_trans, 128 * 128 * sizeof(nv_bfloat16));
        cudaMalloc(&d_x, M * N * sizeof(nv_bfloat16));
        cudaMalloc(&d_y, M * N * sizeof(nv_bfloat16));

        cudaMemcpy(d_trans, h_trans, 128 * 128 * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, M * N * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);

        group_transform_128(d_y, d_trans, d_x, M, N, false);
        cudaDeviceSynchronize();

        cudaMemcpy(h_y, d_y, M * N * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

        if (matrices_equal(h_y, h_expected, M * N)) {
            std::cout << "✓ Test 2 PASSED" << std::endl;
        } else {
            std::cout << "✗ Test 2 FAILED" << std::endl;
        }

        delete[] h_trans;
        delete[] h_x;
        delete[] h_y;
        delete[] h_expected;
        cudaFree(d_trans);
        cudaFree(d_x);
        cudaFree(d_y);
        std::cout << std::endl;
    }

    // Test 3: Scale transform (trans = 2*I, should have y = 2*x)
    {
        std::cout << "=== Test 3: Scale Transform ===" << std::endl;
        int M = 384, N = 128;
        float scale = 2.0f;

        nv_bfloat16 *h_trans, *h_x, *h_y, *h_expected;
        nv_bfloat16 *d_trans, *d_x, *d_y;

        h_trans = new nv_bfloat16[128 * 128];
        h_x = new nv_bfloat16[M * N];
        h_y = new nv_bfloat16[M * N];
        h_expected = new nv_bfloat16[M * N];

        // Initialize trans as scale * identity
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < 128; j++) {
                float val = (i == j) ? scale : 0.0f;
                h_trans[i * 128 + j] = float_to_bfloat16(val);
            }
        }

        init_random_matrix(h_x, M, N, gen);

        // Expected: scale * x
        for (int i = 0; i < M * N; i++) {
            h_expected[i] = float_to_bfloat16(scale * bfloat16_to_float(h_x[i]));
        }

        cudaMalloc(&d_trans, 128 * 128 * sizeof(nv_bfloat16));
        cudaMalloc(&d_x, M * N * sizeof(nv_bfloat16));
        cudaMalloc(&d_y, M * N * sizeof(nv_bfloat16));

        cudaMemcpy(d_trans, h_trans, 128 * 128 * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, M * N * sizeof(nv_bfloat16), cudaMemcpyHostToDevice);

        group_transform_128(d_y, d_trans, d_x, M, N, false);
        cudaDeviceSynchronize();

        cudaMemcpy(h_y, d_y, M * N * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

        if (matrices_equal(h_y, h_expected, M * N, 1e-2)) {
            std::cout << "✓ Test 3 PASSED" << std::endl;
        } else {
            std::cout << "✗ Test 3 FAILED" << std::endl;
        }

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
