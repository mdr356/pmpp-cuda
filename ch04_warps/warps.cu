// PMPP Chapter 4 — Warps, control divergence, and occupancy
//
// build:  nvcc -I common -o warps ch04_warps/warps.cu && ./warps; echo "exit=$?"
// check:  compute-sanitizer ./warps
//
// PROBLEM
// -------
// Chapter 4 has no "write this kernel" exercise; it is about how the hardware
// runs the kernels you already wrote. So this problem is three small
// experiments that make the chapter's claims measurable on your own GPU.
//
// PART A: Device query.
//   Use cudaGetDeviceProperties and print, for device 0:
//     name, SM count, warp size, max threads per block, max threads per SM,
//     max blocks per SM, registers per SM, shared memory per SM, clock rate.
//   Then compute and print, from those numbers:
//     max resident warps per SM, and max resident threads on the whole GPU.
//
// PART B: Control divergence.
//   Two kernels do the SAME total work on N floats. Half the threads run
//   pathA(), half run pathB():
//     divergentKernel : thread i takes path A if (i % 2 == 0), else B.
//                       Every warp contains both paths.
//     uniformKernel   : thread i takes path A if ((i / 32) % 2 == 0), else B.
//                       Every warp takes exactly one path.
//   Time both with cudaEvents. Verify both against the CPU reference.
//
// PART C: Block-size sweep.
//   Launch sweepKernel (provided, trivially memory-bound) over N elements
//   with block sizes 32, 64, 128, 256, 512, 1024. For each:
//     - time the kernel
//     - call cudaOccupancyMaxActiveBlocksPerMultiprocessor and print the
//       theoretical occupancy = (blocks * blockSize / 32) / maxWarpsPerSM
//   Print one line per block size.
//
// Rules:
//   1. Every CUDA call through CUDA_CHECK; launches followed by
//      cudaGetLastError(). Time ONLY the kernel, not the copies.
//   2. Run each timed kernel once untimed first (warm-up), then time it.
//   3. Both Part B kernels must print PASS against the CPU reference.
//   4. compute-sanitizer must be clean.
//   5. Do NOT change pathA/pathB or HEAVY. The compiler must see two
//      genuinely different code paths or there is nothing to diverge.
//
// Questions to answer in a comment at the bottom of this file once it works:
//   Q1. From Part A: how many warps can be resident on one SM? For 256-thread
//       blocks, how many blocks fit per SM by the thread limit? By the block
//       limit? Which one binds on your GPU? Repeat for 64-thread blocks.
//   Q2. Go back to ch02_vecadd (N = 1000, 256-thread blocks). How many warps
//       are in the grid? How many of them have control divergence on the
//       `if (i < N)` guard? What is the SIMD efficiency of that warp
//       (active lanes / 32)? What is the SIMD efficiency of the grid overall?
//   Q3. From Part B: how much slower was divergentKernel than uniformKernel?
//       Explain what the hardware does at the branch in each case. Why is the
//       slowdown not exactly 2x?
//   Q4. From Part C: which block size was fastest, and by how much? Did the
//       theoretical occupancy predict the ordering? At what point did adding
//       occupancy stop helping, and why does a memory-bound kernel not need
//       100% occupancy to reach full throughput?
//   Q5. (Book-style) A kernel uses 48 registers per thread and no shared
//       memory, launched with 512-thread blocks. Using YOUR Part A numbers,
//       which resource limits the number of resident blocks per SM, how many
//       blocks is that, and what occupancy does it give?

#include <cuda_runtime.h>
#include "cuda_check.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N      (1 << 24)   // 16M elements
#define HEAVY  256         // iterations inside each path; keep as is

// ---------------------------------------------------------------------------
// Provided: the two code paths. Same iteration count, different arithmetic,
// so the compiler cannot fold them into one predicated loop. Marked
// __host__ __device__ so the CPU reference runs the identical code.
// ---------------------------------------------------------------------------
__host__ __device__ inline float pathA(float x)
{
    for (int k = 0; k < HEAVY; k++) x = x * 1.0001f + 0.5f;
    return x;
}

__host__ __device__ inline float pathB(float x)
{
    for (int k = 0; k < HEAVY; k++) x = sqrtf(x * x + 0.25f);
    return x;
}

// ---------------------------------------------------------------------------
// YOUR PART B: the two kernels.
//   in, out : N floats
//   divergentKernel : if (i % 2 == 0) out[i] = pathA(in[i]) else pathB(in[i])
//   uniformKernel   : if ((i / 32) % 2 == 0) ... same two paths
//   Guard i < n.
// ---------------------------------------------------------------------------
__global__
void divergentKernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n ) {
        if (i % 2 == 0) out[i] = pathA(in[i]);
        else out[i] = pathB(in[i]);
    }
}

__global__
void uniformKernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if ((i / 32) % 2 == 0) out[i] = pathA(in[i]);
        else out[i] = pathB(in[i]);
    }
}

// ---------------------------------------------------------------------------
// Provided: Part C kernel. Deliberately trivial so it is memory-bound and
// the only thing that changes between launches is the block size.
// ---------------------------------------------------------------------------
__global__
void sweepKernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

// ---------------------------------------------------------------------------
// Provided: host helpers.
// ---------------------------------------------------------------------------
static void fillRandom(float* A, int n)
{
    for (int i = 0; i < n; i++) A[i] = (float)rand() / (float)RAND_MAX;
}

// CPU reference for Part B. `divergent` selects which path-assignment rule.
static void referenceCPU(const float* in, float* out, int n, bool divergent)
{
    for (int i = 0; i < n; i++) {
        bool takeA = divergent ? (i % 2 == 0) : ((i / 32) % 2 == 0);
        out[i] = takeA ? pathA(in[i]) : pathB(in[i]);
    }
}

static bool verify(const char* label, const float* got, const float* want, int n)
{
    for (int i = 0; i < n; i++) {
        float diff = fabsf(got[i] - want[i]);
        float tol  = 1e-3f * fmaxf(1.0f, fabsf(want[i]));
        if (diff > tol) {
            printf("%s: FAIL at %d: got %f want %f\n", label, i, got[i], want[i]);
            return false;
        }
    }
    printf("%s: PASS\n", label);
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
    // -----------------------------------------------------------------------
    // YOUR PART A: device query.
    //   cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    //   Print the fields listed in the header. Keep maxWarpsPerSM in a
    //   variable; Part C needs it.
    // -----------------------------------------------------------------------
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;   

    printf("maxWarpsPerSM = %d\n", maxWarpsPerSM);

    // -----------------------------------------------------------------------
    // Shared setup for Parts B and C.
    // -----------------------------------------------------------------------
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
    // YOUR PART B: divergence.
    //   For each of divergentKernel, uniformKernel (256-thread blocks):
    //     warm-up launch, then record start, launch, record stop
    //     copy d_out -> h_out, referenceCPU(..., divergent = true/false)
    //     ok &= verify(...)
    //     print "<name>  <ms> ms"
    //   Finally print the ratio divergent / uniform.
    // -----------------------------------------------------------------------

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // warm up, first launch we do not measure time in first launch
    divergentKernel<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    // record start, launch kerne and record stop
    CUDA_CHECK(cudaEventRecord(start, 0));
    divergentKernel<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));

    CUDA_CHECK(cudaGetLastError());
    float msDiv = elapsedMs(start, stop);
    printf("divergent %.3f ms\n", msDiv);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    referenceCPU(h_in, h_ref, N, true);
    ok &= verify("divergent",h_out, h_ref, N);

    // warm up for uniform kernel
    uniformKernel<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(start, 0));
    uniformKernel<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));

    CUDA_CHECK(cudaGetLastError());
    float msUni = elapsedMs(start, stop);
    printf("uniform %.3f ms\n", msUni);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    referenceCPU(h_in, h_ref, N, false);
    ok &= verify("uniform",h_out, h_ref, N);

    printf("divergent / uniform = %.3f\n", msDiv / msUni);

    // -----------------------------------------------------------------------
    // YOUR PART C: block-size sweep with sweepKernel.
    //   for blockSize in {32, 64, 128, 256, 512, 1024}:
    //     grid = ceil(N / blockSize)
    //     warm-up, then time one launch
    //     int blocksPerSM;
    //     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM,
    //                                                   sweepKernel, blockSize, 0);
    //     occupancy = (blocksPerSM * blockSize / 32) / (float)maxWarpsPerSM
    //     print "block=%4d  blocks/SM=%2d  occupancy=%5.1f%%  %.3f ms"
    // -----------------------------------------------------------------------

    for (int blockSize = 32; blockSize <= 1024; blockSize *= 2) {
        dim3 block(blockSize);
        dim3 grid((N + block.x - 1) / block.x);

        // warm up
        sweepKernel<<<grid, block>>>(d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(start, 0));
        sweepKernel<<<grid, block>>>(d_in, d_out, N);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaGetLastError());

        int blocksPerSM;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM,
                                                      sweepKernel, blockSize, 0));
        float occupancy = (blocksPerSM * blockSize / 32) / (float)maxWarpsPerSM;
        float msSweep = elapsedMs(start, stop);
        printf("block=%4d  blocks/SM=%2d  occupancy=%5.1f%%  %.3f ms\n",
               blockSize, blocksPerSM, occupancy * 100.0f, msSweep);
    }

    // -----------------------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);
    return ok ? 0 : 1;
}


//   Q1. From Part A: how many warps can be resident on one SM? For 256-thread
//       blocks, how many blocks fit per SM by the thread limit? By the block
//       limit? Which one binds on your GPU? Repeat for 64-thread blocks.
/*
 * Device:  Tesla T4 GPU,
 *  Max warps per SM: 32 
 * for a 256 thread block, with a maximum of 2048 threads per SM 
 *    => 2048 / 256 = 8 blocks per SM
 * Threads limits binds first.
 * 
 * for a 64 threads block, with a maximum of 2048 threads per SM
 *    => 2048 / 64 = 32 blocks per SM
 */

//   Q2. Go back to ch02_vecadd (N = 1000, 256-thread blocks). How many warps
//       are in the grid? How many of them have control divergence on the
//       `if (i < N)` guard? What is the SIMD efficiency of that warp
//       (active lanes / 32)? What is the SIMD efficiency of the grid overall?

/*
 * since one warp is 32 threads, and we can run 256-thread blocks, 
 * therefore each 256-thread block contains 256/32 = 8 warps. 
 * 
 * with N = 1000, the grid contains 1000 / 256 = 3.90625 ~ 4 blocks.
 * each block has 8 warps, so the total number of warps in the grid is 4 * 8 = 32 warps.
 * 
 * SIMD efficiency is (active lanes / 32). 
 * 256+256+256+232 = 1000 active threads
 * the last thread is 232 so we have wastefull thread in the last thread
 * therefore the SIMD efficiency of the last warp is 232 / 256 = 0.90625
 * the overall SIMD efficiency of the grid is 1000 / (4 * 256) = 1000 / 1024 = 0.9765625
 */
//   Q3. From Part B: how much slower was divergentKernel than uniformKernel?
//       Explain what the hardware does at the branch in each case. Why is the
//       slowdown not exactly 2x?

/*
 * divergentKernel was slower than uniformKernel because of branch divergence.
 * When threads within a warp take different execution paths, the hardware serializes
 * the execution of each path, effectively reducing parallelism. The slowdown is not
 * exactly 2x because not all warps experience full divergence and some instructions
 * can still be executed in parallel.
*/

//   Q4. From Part C: which block size was fastest, and by how much? Did the
//       theoretical occupancy predict the ordering? At what point did adding
//       occupancy stop helping, and why does a memory-bound kernel not need
//       100% occupancy to reach full throughput?

/*
 * The fastest block size and its performance relative to others would be determined
 * by the balance between occupancy and memory throughput. The theoretical occupancy
 * might predict the ordering to some extent, but for memory-bound kernels, achieving
 * 100% occupancy is not necessary to reach full throughput because memory latency
 * can be hidden with fewer active warps.
 */

//   Q5. (Book-style) A kernel uses 48 registers per thread and no shared
//       memory, launched with 512-thread blocks. Using YOUR Part A numbers,
//       which resource limits the number of resident blocks per SM, how many
//       blocks is that, and what occupancy does it give?

/*
 * To determine which resource limits the number of resident blocks per SM, we need to
 * consider the number of registers per thread and the block size. Each block has 512
 * threads, and each thread uses 48 registers. The total number of registers per block
 * is 512 * 48 = 24576 registers. If the SM has a limited number of registers (e.g., 65536),
 * the number of resident blocks per SM is limited by the register file. In this case,
 * 65536 / 24576 ≈ 2 blocks per SM. The occupancy is then (2 blocks * 512 threads) / (max
 * threads per SM), which depends on the specific GPU architecture.
 */