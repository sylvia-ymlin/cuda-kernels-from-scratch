#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

// vectorized load and store
#define FLOAT4(a) *(float4*)(&(a))
#define CEIL(a,b) ((a + b -1)/(b))


#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line){
    if (err != cudaSuccess){
        // print the error message
        print("[CUDA ERROR] at file %s(line %d):\n %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

__global__ void elementwise_add(float* A, float* B, float* C, int n){
    // get the thread id
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4; // since we are using float4
    if (idx >= n) return;

    float4 tmp_a = FLOAT4(A[idx]);
    float4 tmp_b = FLOAT4(B[idx]);
    float4 tmp_c;
    // unroll the loop, for better performance
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;
    FLOAT4(C[idx]) = tmp_c;
}

__global__ void sigmoid_float4(float4* X, float4* Y, int n){
    // get the thread id
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4; // since we are using float4
    if (idx >= n) return;
    float4 tmp_x = FLOAT4(X[idx]);
    float4 tmp_y;
    // unroll the loop, for better performance
    tmp_y.x = 1.0 / (1.0 + exp(-tmp_x.x));
    tmp_y.y = 1.0 / (1.0 + exp(-tmp_x.y));
    tmp_y.z = 1.0 / (1.0 + exp(-tmp_x.z));
    tmp_y.w = 1.0 / (1.0 + exp(-tmp_x.w));
    FLOAT4(Y[idx]) = tmp_y;
}

__global__ void relu_float4(float4* X, float4* Y, int n){
    // get the thread id
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4; // since we are using float4
    if (idx >= n) return;
    float4 tmp_x = FLOAT4(X[idx]);
    float4 tmp_y;
    // unroll the loop, for better performance
    tmp_y.x = max(0.0, tmp_x.x);
    tmp_y.y = max(0.0, tmp_x.y);
    tmp_y.z = max(0.0, tmp_x.z);
    tmp_y.w = max(0.0, tmp_x.w);
    FLOAT4(Y[idx]) = tmp_y;
}

int main(){
    constexpr int n = 7;
    // allocate memory on the host
    float *A = (float *)malloc(n * sizeof(float));
    float *B = (float *)malloc(n * sizeof(float));
    float *C = (float *)malloc(n * sizeof(float));
    // initialize the arrays
    for (int i = 0; i < n; i++){
        A[i] = i;
        B[i] = i;
    }
    // allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, n * sizeof(float));
    cudaMalloc((void **)&d_B, n * sizeof(float));
    cudaMalloc((void **)&d_C, n * sizeof(float));
    // copy the data from the host to the device
    cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // set the grid and block size
    dim3 block(256);
    dim3 grid(CEIL(CEIL(n, 4), 256));
    // launch the kernel
    elementwise_add<<<grid, block>>>(d_A, d_B, d_C, n);
    
    _cudaCheck(cudaMemcpy(C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost));

    // print all the arrays
    printf("A: ");
    for (int i = 0; i < n; i++){
        printf("%f ", A[i]);
    }
    printf("\n");
    printf("B: ");
    for (int i = 0; i < n; i++){
        printf("%f ", B[i]);
    }
    printf("\n");
    printf("C: ");
    for (int i = 0; i < n; i++){
        printf("%f ", C[i]);
    }
    printf("\n");

    // free the memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
