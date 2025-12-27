#pragma once

#include <cuda_runtime.h>

__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const int BLOCK_SIZE = 32;                                            
  const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
  const int col = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;

  if (row < M && col < N) {
    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}
