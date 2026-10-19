// PMPP Chapter 4 — __syncthreads(): barrier synchronization within a block
//
// build:  nvcc -I common -o syncthreads ch04_syncthreads/syncthreads.cu && ./syncthreads; echo "exit=$?"
// check:  compute-sanitizer ./syncthreads
// race:   compute-sanitizer --tool racecheck ./syncthreads
//
// PROBLEM
// -------
// Section 4.3 introduces __syncthreads(): a barrier that no thread in a block
// passes until every thread in the block has reached it. It exists so that
// threads can hand data to each other through shared memory. Thread A writes
// a slot, the barrier guarantees the write is finished and visible, then
// thread B reads it. Without the barrier there is no ordering between two
// different threads at all, even inside one block.
//
// This problem makes three things measurable:
//   1. what goes wrong without the barrier (and how to catch it),
//   2. why the barrier must sit outside any divergent branch,
//   3. what a barrier costs.
//
// PART A: Block reverse through shared memory.
//   Each block owns a contiguous slice of `in`. Reverse that slice into `out`:
//     out[base + t] = in[base + (len - 1 - t)]
//   where base = blockIdx.x * blockDim.x and len is the number of valid
//   elements in this block (blockDim.x, or fewer in the last block since N is
//   NOT a multiple of the block size).
//
//   Do it in two phases through a __shared__ tile:
//     phase 1: tile[t] = in[base + t]            (guarded by base + t < n)
//              __syncthreads()
//     phase 2: out[base + t] = tile[len - 1 - t] (guarded by t < len)
//
//   Write it twice:
//     reverseKernel          : with the barrier. Must PASS.
//     reverseNoBarrierKernel : identical, barrier deleted. Expected to be
//                              WRONG. Whether it actually fails on your GPU
//                              is up to the warp scheduler; racecheck will
//                              flag it either way.
//
// PART B: Barrier cost.
//   barrierCostKernel(in, out, n, nBarriers):
//     x = in[i]; repeat nBarriers times { x = x * 1.0001f + 0.5f; __syncthreads(); }
//     out[i] = x
//   The arithmetic is the same whether or not the barrier is there, so the
//   difference in time between nBarriers = 0 and nBarriers = K is K barriers.
//   Time it for nBarriers in {0, 1, 16, 64, 256} at block size 128 and
//   again at block size 1024. Verify every launch against the CPU reference.
//   Report the per-barrier cost per block in nanoseconds:
//     (t[256] - t[0]) / 256 / blocksPerSM_resident  ... or simply the slope
//     (t[256] - t[0]) / 256 as "ns per barrier per kernel"; both are fine,
//     say which one you report.
//
// Rules:
//   1. Every CUDA call through CUDA_CHECK; launches followed by
//      cudaGetLastError(). Time ONLY the kernel, not the copies.
//   2. Run each timed kernel once untimed first (warm-up), then time it.
//   3. __syncthreads() is NEVER inside an `if` that some threads of the block
//      skip. The guards in Part A protect the loads and stores, not the
//      barrier. (Q1 asks why.)
//   4. reverseKernel and every barrierCostKernel launch must print PASS.
//      reverseNoBarrierKernel is allowed to FAIL; it is the point.
//   5. compute-sanitizer (memcheck) must be clean. racecheck must report
//      hazards for reverseNoBarrierKernel and NONE for reverseKernel.
//   6. Do NOT make N a multiple of the block size. The partial last block is
//      what forces the barrier-vs-guard question.
//
// Questions to answer in a comment at the bottom of this file once it works:
//   Q1. Why must __syncthreads() sit outside the `if (base + t < n)` guard?
//       In the last block some threads are past the end of the array. What
//       does the CUDA programming guide say happens if those threads skip
//       the barrier? What did your GPU actually do when you tried it?
//   Q2. Run racecheck on both Part A kernels. Paste the hazard line for
//       reverseNoBarrierKernel. Which shared address, and which two threads
//       (and therefore which two warps) does it name? Explain why thread 0
//       is the one most likely to read stale data, and why the no-barrier
//       kernel can still print PASS on some runs.
//   Q3. From Part B: what is the cost of one barrier in ns, and in SM clock
//       cycles (clock rate is in cudaDeviceProp)? Did it change between
//       block size 128 and 1024? Compare it with one global memory load
//       (hundreds of cycles). What does this tell you about how much work
//       should sit between barriers in Chapter 5's tiled matmul?
//   Q4. __syncthreads() synchronizes a block. There is no equivalent single
//       instruction for the whole grid. Using sections 4.2 (block
//       scheduling) and 4.7 (resource partitioning), explain why a grid-wide
//       barrier would break transparent scalability, and what the book says
//       to do when you need one.
//   Q5. (Book-style, previews Chapter 5) Tiled matmul has TWO barriers per
//       tile iteration: one after loading the tile, one after computing with
//       it. Name the hazard each one prevents (read-after-write or
//       write-after-read), and say what would go wrong if each were removed.

#include <cuda_runtime.h>
#include "cuda_check.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N          ((1 << 22) + 37)   // 4M + 37: last block is partial on purpose
#define TILE_MAX   1024               // largest block size we will launch with

// ---------------------------------------------------------------------------
// YOUR PART A: the two reverse kernels.
//   base = blockIdx.x * blockDim.x
//   len  = min(blockDim.x, n - base)      valid elements in this block
//   tile is sized TILE_MAX so one kernel works for any block size <= 1024.
// ---------------------------------------------------------------------------
__global__
void reverseKernel(const float* in, float* out, int n)
{
    __shared__ float tile[TILE_MAX];
    // TODO: phase 1 (guarded load), barrier, phase 2 (guarded reversed store)
    (void)tile; (void)in; (void)out; (void)n;
}

__global__
void reverseNoBarrierKernel(const float* in, float* out, int n)
{
    __shared__ float tile[TILE_MAX];
    // TODO: same as reverseKernel with the __syncthreads() removed.
    (void)tile; (void)in; (void)out; (void)n;
}

// ---------------------------------------------------------------------------
// YOUR PART B: the barrier-cost kernel.
//   Guard i < n around the load and the store. The loop and the barrier run
//   for EVERY thread, including the ones past n, otherwise rule 3 is broken.
// ---------------------------------------------------------------------------
__global__
void barrierCostKernel(const float* in, float* out, int n, int nBarriers)
{
    // TODO
    (void)in; (void)out; (void)n; (void)nBarriers;
}

// ---------------------------------------------------------------------------
// Provided: host helpers.
// ---------------------------------------------------------------------------
static void fillRandom(float* A, int n)
{
    for (int i = 0; i < n; i++) A[i] = (float)rand() / (float)RAND_MAX;
}

// CPU reference for Part A: reverse each blockSize-wide slice in place order.
static void referenceReverse(const float* in, float* out, int n, int blockSize)
{
    for (int base = 0; base < n; base += blockSize) {
        int len = (n - base < blockSize) ? (n - base) : blockSize;
        for (int t = 0; t < len; t++) out[base + t] = in[base + (len - 1 - t)];
    }
}

// CPU reference for Part B: the same arithmetic, no barriers to wait on.
static void referenceBarrierCost(const float* in, float* out, int n, int nBarriers)
{
    for (int i = 0; i < n; i++) {
        float x = in[i];
        for (int k = 0; k < nBarriers; k++) x = x * 1.0001f + 0.5f;
        out[i] = x;
    }
}

static bool verify(const char* label, const float* got, const float* want, int n)
{
    for (int i = 0; i < n; i++) {
        float diff = fabsf(got[i] - want[i]);
        float tol  = 1e-4f * fmaxf(1.0f, fabsf(want[i]));
        if (diff > tol) {
            printf("%-28s FAIL at %d: got %f want %f\n", label, i, got[i], want[i]);
            return false;
        }
    }
    printf("%-28s PASS\n", label);
    return true;
}

// Returns elapsed ms between two recorded events, synchronising on `stop`.
static float elapsedMs(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  (%d SMs, %d kHz)\n\n", prop.name, prop.multiProcessorCount, prop.clockRate);

    const size_t bytes = (size_t)N * sizeof(float);
    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    if (!h_in || !h_out || !h_ref) { fprintf(stderr, "malloc failed\n"); return 1; }
    srand(42);
    fillRandom(h_in, N);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    bool ok = true;

    // -----------------------------------------------------------------------
    // YOUR PART A: block reverse, 256-thread blocks.
    //   reverseKernel          -> verify, ok &= result
    //   reverseNoBarrierKernel -> verify, print result but do NOT fold it
    //                             into `ok` (it is allowed to fail)
    //   Clear d_out (cudaMemset) before each launch so a stale correct
    //   answer from the previous kernel cannot mask a bug.
    // -----------------------------------------------------------------------
    {
        const int blockSize = 256;
        // TODO
        (void)blockSize;
    }

    // -----------------------------------------------------------------------
    // YOUR PART B: barrier cost sweep.
    //   for blockSize in {128, 1024}:
    //     for nBarriers in {0, 1, 16, 64, 256}:
    //       warm-up, then time one launch
    //       copy back, referenceBarrierCost, ok &= verify
    //       print "block=%4d  barriers=%3d  %.3f ms"
    //     print the slope (t[256] - t[0]) / 256 in ns, and in cycles
    //     using prop.clockRate (kHz).
    // -----------------------------------------------------------------------
    {
        const int blockSizes[] = {128, 1024};
        const int counts[]     = {0, 1, 16, 64, 256};
        // TODO
        (void)blockSizes; (void)counts;
    }

    // -----------------------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// ANSWERS (fill in once the code works and has been run on a GPU)
// ---------------------------------------------------------------------------
//   Q1.
//
//   Q2.
//
//   Q3.
//
//   Q4.
//
//   Q5.
