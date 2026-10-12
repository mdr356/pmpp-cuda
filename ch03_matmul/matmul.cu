// PMPP Chapter 3 — Matrix multiplication (naive, one thread per output element)
//
// build:  nvcc -I common -o matmul ch03_matmul/matmul.cu && ./matmul; echo "exit=$?"
// check:  compute-sanitizer ./matmul
//
// PROBLEM
// -------
// Compute P = M * N for square WIDTH x WIDTH matrices of floats, stored
// row-major (element (row, col) lives at index row*WIDTH + col).
//
//   P[row][col] = sum over k of  M[row][k] * N[k][col]
//
// Rules:
//   1. One thread per OUTPUT element of P. 2-D grid, 2-D blocks.
//      Thread (x, y) computes P[row = y][col = x].
//   2. Each thread loops k = 0 .. WIDTH-1, accumulating in a float register,
//      then writes its single result to P. No shared memory yet (Chapter 5).
//   3. Guard against threads that fall outside WIDTH x WIDTH. Pick a WIDTH
//      that is NOT a multiple of your block size at least once to prove it.
//   4. Do not read or write out of bounds (compute-sanitizer must be clean).
//   5. verify() compares against a CPU reference. It must print "PASS".
//
// Questions to answer in a comment at the bottom of this file once it works:
//   Q1. How many global-memory LOADS does one thread do? How many STOREs?
//       How many floating-point operations? What is the FLOP-per-byte ratio,
//       and what does that say about whether this kernel is compute-bound or
//       memory-bound? (Chapter 5 calls this the compute-to-global-memory-
//       access ratio and fixes it with tiling.)
//   Q2. Threads in the same block that share a ROW of P all read the same
//       row of M. Threads that share a COLUMN of P all read the same column
//       of N. How many times is each element of M loaded from global memory
//       across the whole grid? Each element of N?
//   Q3. In one loop iteration k, look at 32 consecutive threads along x
//       (same row, cols c..c+31). Which addresses do they read from M? From
//       N? Which of the two patterns do you expect the hardware to like
//       better, and why? (Chapter 6 names this "coalescing".)
//   Q4. Report the GFLOP/s this program prints for WIDTH = 512 and 1024 and
//       compare it to your GPU's peak FP32 throughput. What fraction are you
//       achieving? Keep this number; you will beat it in Chapter 5.
//   Q5. (Book exercise 3.1) Sketch, in words, a variant where each thread
//       computes one whole ROW of P, and one where each thread computes one
//       whole COLUMN. What are the pros and cons of each versus one thread
//       per element?

#include <cuda_runtime.h>
#include "cuda_check.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 512   // try 512, 1024, and something odd like 500

// ---------------------------------------------------------------------------
// YOUR PART 1: the kernel.
//   M, N, P : WIDTH*WIDTH floats each, row-major
//   Compute row/col from block and thread indices, guard, loop over k,
//   accumulate, store once.
// ---------------------------------------------------------------------------
__global__
void matMulKernel(const float* M, const float* N, float* P, int width)
{
    // TODO
}

// ---------------------------------------------------------------------------
// Host helpers (provided).
// ---------------------------------------------------------------------------
static void fillRandom(float* A, int n)
{
    for (int i = 0; i < n; i++) A[i] = (float)rand() / (float)RAND_MAX;
}

static void matMulCPU(const float* M, const float* N, float* P, int width)
{
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) sum += M[row * width + k] * N[k * width + col];
            P[row * width + col] = sum;
        }
    }
}

static bool verify(const float* got, const float* want, int n)
{
    // Summing WIDTH products in a different order gives slightly different
    // rounding on CPU vs GPU, so compare with a relative tolerance.
    for (int i = 0; i < n; i++) {
        float diff = fabsf(got[i] - want[i]);
        float tol  = 1e-4f * fmaxf(1.0f, fabsf(want[i]));
        if (diff > tol) {
            printf("FAIL at %d (row %d, col %d): got %f want %f\n",
                   i, i / WIDTH, i % WIDTH, got[i], want[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

int main()
{
    const int    n     = WIDTH * WIDTH;
    const size_t bytes = (size_t)n * sizeof(float);

    float* h_M   = (float*)malloc(bytes);
    float* h_N   = (float*)malloc(bytes);
    float* h_P   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    if (!h_M || !h_N || !h_P || !h_ref) { fprintf(stderr, "malloc failed\n"); return 1; }

    srand(42);
    fillRandom(h_M, n);
    fillRandom(h_N, n);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -----------------------------------------------------------------------
    // YOUR PART 2: device side.
    //   cudaMalloc d_M / d_N / d_P (bytes each)
    //   copy h_M, h_N to the device
    //   pick a 2-D block, compute a 2-D grid that covers WIDTH x WIDTH
    //   record `start`, launch matMulKernel, record `stop`
    //   check launch + execution errors
    //   copy d_P back into h_P
    //   free device memory
    // -----------------------------------------------------------------------

    // TODO

    // -----------------------------------------------------------------------

    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double flops = 2.0 * (double)WIDTH * (double)WIDTH * (double)WIDTH;  // one mul + one add per k
    printf("WIDTH=%d  kernel %.3f ms  %.1f GFLOP/s\n", WIDTH, ms, flops / (ms * 1e6));

    matMulCPU(h_M, h_N, h_ref, WIDTH);
    bool ok = verify(h_P, h_ref, n);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_M); free(h_N); free(h_P); free(h_ref);
    return ok ? 0 : 1;
}
