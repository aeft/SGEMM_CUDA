#pragma once

#include <cuda_runtime.h>

/*

Naive SGEMM kernel implementation
 
TODO: Implement the naive matrix multiplication kernel

Each thread computes one element of the output matrix C
C[i][j] = alpha * A[i][:] @ B[:][j] + beta * C[i][j]

Matrix sizes: MxK * KxN = MxN

*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // TODO: Calculate row and column index
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: Bounds check
    if (row < M && col < N) {
        float sum = 0.0f;
    
        // TODO: Compute dot product
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
    
        // TODO: Write result
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
