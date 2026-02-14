#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "src_practice/runner.cuh"

// Simple matrix initialization helpers
void initMatrix(float* mat, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = value;
    }
}

void initMatrixIncremental(float* mat, int rows, int cols, double x) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)i*x;
    }
}

void printMatrix(const char* name, float* mat, int rows, int cols, int maxRows = 8, int maxCols = 8) {
    printf("%s (%dx%d):\n", name, rows, cols);
    int printRows = (rows < maxRows) ? rows : maxRows;
    int printCols = (cols < maxCols) ? cols : maxCols;
    for (int i = 0; i < printRows; i++) {
        for (int j = 0; j < printCols; j++) {
            printf("%6.1f ", mat[i * cols + j]);
        }
        if (cols > maxCols) printf("...");
        printf("\n");
    }
    if (rows > maxRows) printf("...\n");
    printf("\n");
}

int main() {
    // Configure matrix size - use small values for debugging
    const int M = 128;
    const int N = 128;
    const int K = 128;

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices - customize as needed
    initMatrixIncremental(h_A, M, K, 1.0);
    initMatrixIncremental(h_B, K, N, 2.0);
    initMatrix(h_C, M, N, 0.0f);      

    // Print input matrices
    // printMatrix("A", h_A, M, K);
    // printMatrix("B", h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel via runner function
    // kernel_num: 4=1D Blocktiling, 5=2D Blocktiling, etc.
    int kernel_num = 5;
    run_kernel(kernel_num, M, N, K, 1.0f, d_A, d_B, 0.0f, d_C, nullptr);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    // printMatrix("C", h_C, M, N);

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
