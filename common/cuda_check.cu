#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

inline void CUDA_CHECK(cudaError_t call) {
    if(call != cudaSuccess) {
        fprintf(stderr, "cuda Runtime error %s: %s\n", cudaGetErrorString(call), cudaGetErrorName(call));
        abort();
    }
}