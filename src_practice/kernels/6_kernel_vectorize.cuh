#pragma once

#include <cuda_runtime.h>

// BM: Block tile size in M dimension (rows of A/C)
// BN: Block tile size in N dimension (cols of B/C)
// BK: Block tile size in K dimension (shared dimension)
// TM: Thread tile size (elements per thread in M dimension)
// TN: Thread tile size (elements per thread in N dimension)
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void SgemmVectorize(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C) {
  int threadPerRow = BN / TN;
  int threadRowInBlock = threadIdx.x / threadPerRow;
  int threadColInBlock = threadIdx.x % threadPerRow;

  __shared__ float sharedA[BK * BM]; // transposed tile of A in shared memory
  __shared__ float sharedB[BK * BN];

  float localC[TM][TN] = {0.0};
  for (int kBlockStart = 0; kBlockStart < K; kBlockStart += BK) {
    // Load to sharedA
    // Load 4 elements of A and transpose them while storing to shared memory
    int threadPerRowLoadA = BK / 4;
    int loadRowA = threadIdx.x / threadPerRowLoadA;
    int loadColA = threadIdx.x % threadPerRowLoadA;
    int rowA = blockIdx.x * BM + loadRowA;
    int colA = kBlockStart + loadColA * 4;
    float4 tmpA = reinterpret_cast<const float4 *>(&A[rowA * K + colA])[0];
    sharedA[(loadColA * 4 + 0) * BM + loadRowA] =
        tmpA.x; // Transpose while storing
    sharedA[(loadColA * 4 + 1) * BM + loadRowA] = tmpA.y;
    sharedA[(loadColA * 4 + 2) * BM + loadRowA] = tmpA.z;
    sharedA[(loadColA * 4 + 3) * BM + loadRowA] = tmpA.w;
    // Load to sharedB
    int threadPerRowLoadB = BN / 4;
    int rowB = kBlockStart + (threadIdx.x / threadPerRowLoadB);
    int colB = blockIdx.y * BN + (threadIdx.x % threadPerRowLoadB) * 4;
    float4 tmpB = reinterpret_cast<const float4 *>(&B[rowB * N + colB])[0];
    reinterpret_cast<float4 *>(&sharedB[threadIdx.x * 4])[0] = tmpB;
    __syncthreads();
    for (int tmIdx = 0; tmIdx < TM; tmIdx++) {   // iterate over thread tile (M)
      for (int tnIdx = 0; tnIdx < TN; tnIdx++) { // iterate over thread tile (N)
        for (int kIdx = 0; kIdx < BK; kIdx++) {  // iterate over K dimension
          int sharedARow = threadRowInBlock * TM + tmIdx;
          int sharedBCol = threadColInBlock * TN + tnIdx;
          localC[tmIdx][tnIdx] +=
              sharedA[kIdx * BM + sharedARow] * sharedB[kIdx * BN + sharedBCol];
        }
      }
    }
    __syncthreads();
  }
  for (int tmIdx = 0; tmIdx < TM; tmIdx++) {
    for (int tnIdx = 0; tnIdx < TN; tnIdx += 4) {
      int rowC = blockIdx.x * BM + threadRowInBlock * TM + tmIdx;
      int colC = blockIdx.y * BN + threadColInBlock * TN + tnIdx;
      float4 tmpC;
      tmpC = reinterpret_cast<const float4 *>(&C[rowC * N + colC])[0];
      tmpC.x = alpha * localC[tmIdx][tnIdx] + beta * tmpC.x;
      tmpC.y = alpha * localC[tmIdx][tnIdx + 1] + beta * tmpC.y;
      tmpC.z = alpha * localC[tmIdx][tnIdx + 2] + beta * tmpC.z;
      tmpC.w = alpha * localC[tmIdx][tnIdx + 3] + beta * tmpC.w;
      reinterpret_cast<float4 *>(&C[rowC * N + colC])[0] = tmpC;
    }
  }
}
