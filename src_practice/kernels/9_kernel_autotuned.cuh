#pragma once

#include <cuda_runtime.h>

// BM: Block tile size in M dimension (rows of A/C)
// BN: Block tile size in N dimension (cols of B/C)
// BK: Block tile size in K dimension (shared dimension)
// TM: Thread tile size (elements per thread in M dimension)
// TN: Thread tile size (elements per thread in N dimension)
template <const int BM, const int BN, const int BK, const int TM, const int TN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    SgemmAutotuned(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C) {

  const int WM = TM * 16, WN = TN * 16;
  const int WMITER = BM / WM, WNITER = BN / WN;

  const int threadPerRow = WN / TN;
  int threadRowInBlock = threadIdx.x / threadPerRow;
  int threadColInBlock = threadIdx.x % threadPerRow;

  __shared__ float sharedA[BK * BM]; // transposed tile of A in shared memory
  __shared__ float sharedB[BK * BN];

  float localC[TM * WMITER][TN * WNITER] = {0.0};
  for (int kBlockStart = 0; kBlockStart < K; kBlockStart += BK) {
    // Load to sharedA
    const int loadCntM = BM * BK / 4;
    for (int l = 0; l < loadCntM; l += NUM_THREADS) {
      int threadPerRowLoadA = BK / 4;
      int loadRowA = (l + threadIdx.x) / threadPerRowLoadA;
      int loadColA = threadIdx.x % threadPerRowLoadA;
      int rowA = blockIdx.x * BM + loadRowA;
      int colA = kBlockStart + loadColA * 4;
      float4 tmpA = reinterpret_cast<const float4 *>(&A[rowA * K + colA])[0];
      sharedA[(loadColA * 4 + 0) * BM + loadRowA] =
          tmpA.x; // Transpose while storing
      sharedA[(loadColA * 4 + 1) * BM + loadRowA] = tmpA.y;
      sharedA[(loadColA * 4 + 2) * BM + loadRowA] = tmpA.z;
      sharedA[(loadColA * 4 + 3) * BM + loadRowA] = tmpA.w;
    }
    // Load to sharedB
    const int loadCntN = BN * BK / 4;
    for (int l = 0; l < loadCntN; l += NUM_THREADS) {
      int threadPerRowLoadB = BN / 4;
      int rowB = kBlockStart + ((l + threadIdx.x) / threadPerRowLoadB);
      int colB = blockIdx.y * BN + (threadIdx.x % threadPerRowLoadB) * 4;
      float4 tmpB = reinterpret_cast<const float4 *>(&B[rowB * N + colB])[0];
      reinterpret_cast<float4 *>(&sharedB[(l + threadIdx.x) * 4])[0] = tmpB;
    }
    __syncthreads();
    for (int wmIdx = 0; wmIdx < WMITER; wmIdx++) {
      for (int wnIdx = 0; wnIdx < WNITER; wnIdx++) {
        for (int tmIdx = 0; tmIdx < TM;
             tmIdx++) { // iterate over thread tile (M)
          for (int tnIdx = 0; tnIdx < TN;
               tnIdx++) { // iterate over thread tile (N)
            for (int kIdx = 0; kIdx < BK; kIdx++) { // iterate over K dimension
              int sharedARow = wmIdx * WM + threadRowInBlock * TM + tmIdx;
              int sharedBCol = wnIdx * WN + threadColInBlock * TN + tnIdx;
              localC[wmIdx * TM + tmIdx][wnIdx * TN + tnIdx] +=
                  sharedA[kIdx * BM + sharedARow] *
                  sharedB[kIdx * BN + sharedBCol];
            }
          }
        }
      }
    }
    __syncthreads();
  }
  for (int wmIdx = 0; wmIdx < WMITER; wmIdx++) {
    for (int wnIdx = 0; wnIdx < WNITER; wnIdx++) {
      for (int tmIdx = 0; tmIdx < TM; tmIdx++) {
        for (int tnIdx = 0; tnIdx < TN; tnIdx += 4) {
          int rowC =
              blockIdx.x * BM + wmIdx * WM + threadRowInBlock * TM + tmIdx;
          int colC =
              blockIdx.y * BN + wnIdx * WN + threadColInBlock * TN + tnIdx;
          float4 tmpC;
          tmpC = reinterpret_cast<const float4 *>(&C[rowC * N + colC])[0];
          tmpC.x = alpha * localC[wmIdx * TM + tmIdx][wnIdx * TN + tnIdx] +
                   beta * tmpC.x;
          tmpC.y = alpha * localC[wmIdx * TM + tmIdx][wnIdx * TN + tnIdx + 1] +
                   beta * tmpC.y;
          tmpC.z = alpha * localC[wmIdx * TM + tmIdx][wnIdx * TN + tnIdx + 2] +
                   beta * tmpC.z;
          tmpC.w = alpha * localC[wmIdx * TM + tmIdx][wnIdx * TN + tnIdx + 3] +
                   beta * tmpC.w;
          reinterpret_cast<float4 *>(&C[rowC * N + colC])[0] = tmpC;
        }
      }
    }
  }
}
