#pragma once

#include <cuda_runtime.h>

// BM: Block tile size in M dimension (rows of A/C)
// BN: Block tile size in N dimension (cols of B/C)
// BK: Block tile size in K dimension (shared dimension)
// TM: Thread tile size (elements per thread in M dimension)
// TN: Thread tile size (elements per thread in N dimension)
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void Sgemm2DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  int threadPerRow = BN / TN;
  int threadRowInBlock = threadIdx.x / threadPerRow;
  int threadColInBlock = threadIdx.x % threadPerRow;

  __shared__ float sharedA[BM * BK], sharedB[BK * BN];
  float localC[TM][TN] = {0.0};
  int loadCnt = BM * BK; // Here we assume BM=BN to make data load easier
  int loadPerIter = BM * BN / (TM * TN);
  for (int kBlockStart = 0; kBlockStart < K; kBlockStart += BK) {
    // Load to shared memory
    for (int l = 0; l < loadCnt; l += loadPerIter) {
      // Load to sharedA
      int aRow = blockIdx.x * BM + (l + threadIdx.x) / BK;
      int aCol = kBlockStart + (l + threadIdx.x) % BK;
      sharedA[l + threadIdx.x] = A[aRow * K + aCol];
      // Load to sharedB
      int bRow = kBlockStart + (l + threadIdx.x) / BN;
      int bCol = blockIdx.y * BN + (l + threadIdx.x) % BN;
      sharedB[l + threadIdx.x] = B[bRow * N + bCol];
    }
    __syncthreads();
    for (int tmIdx = 0; tmIdx < TM; tmIdx++) {   // iterate over thread tile (M)
      for (int tnIdx = 0; tnIdx < TN; tnIdx++) { // iterate over thread tile (N)
        for (int kIdx = 0; kIdx < BK; kIdx++) {  // iterate over K dimension
          int sharedARow = threadRowInBlock * TM + tmIdx;
          int sharedBCol = threadColInBlock * TN + tnIdx;
          localC[tmIdx][tnIdx] +=
              sharedA[sharedARow * BK + kIdx] * sharedB[kIdx * BN + sharedBCol];
        }
      }
    }
    __syncthreads();
  }
  for (int tmIdx = 0; tmIdx < TM; tmIdx++) {
    for (int tnIdx = 0; tnIdx < TN; tnIdx++) {
      int cRow = blockIdx.x * BM + threadRowInBlock * TM + tmIdx;
      int cCol = blockIdx.y * BN + threadColInBlock * TN + tnIdx;
      C[cRow * N + cCol] =
          alpha * localC[tmIdx][tnIdx] + beta * C[cRow * N + cCol];
    }
  }
}
