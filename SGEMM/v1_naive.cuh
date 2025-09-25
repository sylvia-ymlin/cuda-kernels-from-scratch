#pragma once // Prevent multiple inclusions of this header file

#include <cstdio> // for printf
#include <cstdlib> // for exit
#include <cublas_v2.h> // cuBLAS header
#include <cuda_runtime.h> // CUDA runtime

/*

Matrix sizes:
MxK * KxN = MxN

*/

// inputs: dims of matrices, alpha, A, B, beta, C
// C = α*(A@B)+β*C
// const and uint: explicitly state that these pointers will not be modified, for code readability and robustness
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;

    // safe check, avoid out-of-bounds access
    if (x < M && y < N) {
        float sum = 0.0;
        // take the row of A and column of B to compute the dot product
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        // the result is written to C at (row, col)
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}