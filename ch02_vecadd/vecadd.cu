#include <cuda_runtime.h>
#include "cuda_check.cuh"
#include <stdio.h>
#include <stdlib.h>

/*
* nvcc -I common -o vecadd ch02/vecadd.cu && ./vecadd; echo "exit=$?"
compute-sanitizer ./vecadd
*/
__global__
void vecAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


static bool verify(float *A, float *B, float *C, int N) {
    for(int i = 0; i < N; i++) {
        if(A[i] + B[i] != C[i]) {
            printf("Verification failed at index %d: %f + %f != %f\n", i, A[i], B[i], C[i]);
            return false;
        }
    }
    printf("Verification passed!\n");
    return true;
}

int main() {
    int N = 1000; // Example size

    // allocate host memory
    float *A_h = static_cast<float*>(malloc(sizeof(float) * N));
    float *B_h = static_cast<float*>(malloc(sizeof(float) * N));
    float *C_h = static_cast<float*>(malloc(sizeof(float) * N));

    // initialize host arrays
    for(int i = 0; i < N; i++) {
        A_h[i] = static_cast<float>(i);
        B_h[i] = static_cast<float>(i);
        C_h[i] = 0.0f;
    }

    // allocate memory on device
    float *A_d, *B_d, *C_d;
    
    const size_t size = static_cast<size_t>(N) * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**) &A_d, size));
    CUDA_CHECK(cudaMalloc((void**) &B_d, size));
    CUDA_CHECK(cudaMalloc((void**) &C_d, size));

    // COPY DATA FROM HOST TO DEVICE 
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice));

    // launch Kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    vecAdd<<<grid, block>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaGetLastError());

    // copy data from device to host
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // verify
    bool ok = verify(A_h, B_h, C_h, N);
       
    // FREE DEVICE MEMORY
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    // free host memory
    free(A_h);
    free(B_h);
    free(C_h);


    return ok ? EXIT_SUCCESS : EXIT_FAILURE;

}
