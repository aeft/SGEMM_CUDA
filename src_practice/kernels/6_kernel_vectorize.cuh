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

  __shared__ float sharedA[BM * BK], sharedB[BK * BN];
  float localC[TM][TN] = {0.0};
  for (int kBlockStart = 0; kBlockStart < K; kBlockStart += BK) {
    // Load to shared memory
    // Load to sharedA
    int threadPerRowLoadA = BK / 4;
    int aRow = blockIdx.x * BM + (threadIdx.x / threadPerRowLoadA);
    int aCol = kBlockStart + (threadIdx.x % threadPerRowLoadA) * 4;
    float4 tmpA = reinterpret_cast<const float4 *>(&A[aRow * K + aCol])[0];
    reinterpret_cast<float4 *>(&sharedA[threadIdx.x * 4])[0] = tmpA;
    // Load to sharedB
    int threadPerRowLoadB = BN / 4;
    int bRow = kBlockStart + (threadIdx.x / threadPerRowLoadB);
    int bCol = blockIdx.y * BN + (threadIdx.x % threadPerRowLoadB) * 4;
    float4 tmpB = reinterpret_cast<const float4 *>(&B[bRow * N + bCol])[0];
    reinterpret_cast<float4 *>(&sharedB[threadIdx.x * 4])[0] = tmpB;
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
    for (int tnIdx = 0; tnIdx < TN; tnIdx += 4) {
      int cRow = blockIdx.x * BM + threadRowInBlock * TM + tmIdx;
      int cCol = blockIdx.y * BN + threadColInBlock * TN + tnIdx;
      float4 tmpC;
      tmpC = reinterpret_cast<const float4 *>(&C[cRow * N + cCol])[0];
      tmpC.x = alpha * localC[tmIdx][tnIdx] + beta * tmpC.x;
      tmpC.y = alpha * localC[tmIdx][tnIdx + 1] + beta * tmpC.y;
      tmpC.z = alpha * localC[tmIdx][tnIdx + 2] + beta * tmpC.z;
      tmpC.w = alpha * localC[tmIdx][tnIdx + 3] + beta * tmpC.w;
      reinterpret_cast<float4 *>(&C[cRow * N + cCol])[0] = tmpC;
    }
  }
}
