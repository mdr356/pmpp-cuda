#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

inline cuda_check_t(cudaError_t call) {
    do {
        if(call != cudaSuccess) {
            fprintf(stderr, "cuda Runtime error %s\n", cudaGetErrorString(call));
            assert(call == cudaSuccess);
        }
    } while(0)
}