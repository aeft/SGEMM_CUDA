#pragma once

#include <cuda_runtime.h>

// BM: Block tile size in M dimension (rows of A/C)
// BN: Block tile size in N dimension (cols of B/C)
// BK: Block tile size in K dimension (shared dimension)
// TM: Thread tile size (elements per thread in M dimension)
template <const int BM, const int BN, const int BK, const int TM>
__global__ void Sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  __shared__ float sharedA[BM * BK], sharedB[BK * BN];
  // Each thread computes TM elements in a column
  float localC[TM] = {0.0};
  // Thread-to-tile mapping: each thread computes a TM-element column in the
  // output tile
  // `threadRowInBlock` gives the thread's row within the thread block
  int threadRowInBlock = threadIdx.x / BN;
  // `threadColInBlock` gives the thread's column within the thread block
  int threadColInBlock = threadIdx.x % BN;

  // Each thread loads one A element at relative position (aRowInBlock,
  // aColInBlock) within the block
  int aRowInBlock = threadIdx.x / BK;
  int aColInBlock = threadIdx.x % BK;
  // Each thread loads one B element at relative position (bRowInBlock,
  // bColInBlock) within the block
  int bRowInBlock = threadIdx.x / BN;
  int bColInBlock = threadIdx.x % BN;

  // iterate over K dimension
  for (int kBlockStart = 0; kBlockStart < K; kBlockStart += BK) {
    // Load data cooperatively: each thread loads one element from A and B
    int aRow = blockIdx.x * BM + aRowInBlock;
    int aCol = kBlockStart + aColInBlock;
    // Boundary check to support TM < 8 (slight performance overhead)
    if (aRow < M) {
      // All threads cooperatively load submatrix A starting at (blockIdx.x*BM,
      // kBlockStart)
      sharedA[threadIdx.x] = A[aRow * K + aCol];
    }
    int bRow = kBlockStart + bRowInBlock;
    int bCol = blockIdx.y * BN + bColInBlock;
    if (bRow < K) {
      // All threads cooperatively load submatrix B starting at (kBlockStart,
      // blockIdx.y*BN)
      sharedB[threadIdx.x] = B[bRow * N + bCol];
    }
    __syncthreads();
    for (int kIdx = 0; kIdx < BK; kIdx++) { // iterate over K dimension
      float tmpB = sharedB[kIdx * BN + threadColInBlock];
      for (int tmIdx = 0; tmIdx < TM; tmIdx++) { // iterate over thread tile
        int sharedARow = threadRowInBlock * TM + tmIdx;
        // Actually, we can use sharedB[...] directly here, since the compiler
        // will automatically allocate a register.
        localC[tmIdx] += sharedA[sharedARow * BK + kIdx] * tmpB;
      }
    }
    __syncthreads();
  }
  for (int tmIdx = 0; tmIdx < TM; tmIdx++) {
    // Determine `cRow`:
    // `blockIdx.x * BM` locates the output block's starting row
    // `threadRowInBlock` gives the thread's row within the thread block
    // `threadRowInBlock * TM` accounts for each thread handling TM elements
    // (i.e., TM rows) `tmIdx` pinpoints the specific element within the tile
    int cRow = blockIdx.x * BM + threadRowInBlock * TM + tmIdx;
    // Determine `cCol`:
    // `blockIdx.y * BN` locates the output block's starting col
    // `threadColInBlock` gives the thread's column within the thread block
    int cCol = blockIdx.y * BN + threadColInBlock;
    C[cRow * N + cCol] = alpha * localC[tmIdx] + beta * C[cRow * N + cCol];
  }
}