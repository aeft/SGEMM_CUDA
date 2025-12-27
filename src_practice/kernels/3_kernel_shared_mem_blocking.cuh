#pragma once

#include <cuda_runtime.h>

__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int BLOCK_SIZE = 32;
  const int threadRow = threadIdx.x / BLOCK_SIZE;
  const int threadCol = threadIdx.x % BLOCK_SIZE;
  const int row = blockIdx.x * BLOCK_SIZE + threadRow;
  const int col = blockIdx.y * BLOCK_SIZE + threadCol;

  __shared__ float sharedA[BLOCK_SIZE * BLOCK_SIZE],
      sharedB[BLOCK_SIZE * BLOCK_SIZE];

  float sum = 0.0f;

  for (int blockL = 0; blockL < K; blockL += BLOCK_SIZE) {
    int aRow = row;
    int aCol = blockL + threadCol;
    // Commenting out the boundary check can make it slightly faster.
    if (aRow < M && aCol < K)
      sharedA[threadIdx.x] = A[aRow * K + aCol];
    else
      sharedA[threadIdx.x] = 0;
    int bRow = blockL + threadRow;
    int bCol = col;
    if (bRow < K && bCol < N)
      sharedB[threadIdx.x] = B[bRow * N + bCol];
    else
      sharedB[threadIdx.x] = 0;
    __syncthreads();
    for (int ik = 0; ik < BLOCK_SIZE; ik++) {
      sum += sharedA[threadRow * BLOCK_SIZE + ik] *
             sharedB[ik * BLOCK_SIZE + threadCol];
    }
    __syncthreads();
  }
  if (row < M && col < N)
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}