// PMPP Chapter 3 — Image blur (box filter)
//
// build:  nvcc -I common -I ch03_grayscale -o blur ch03_blur/blur.cu && ./blur; echo "exit=$?"
// check:  compute-sanitizer ./blur
//
// PROBLEM
// -------
// Take the grayscale image produced by ch03_grayscale (out.png) and blur it.
// Each output pixel is the AVERAGE of all the input pixels inside a square
// window of side (2*BLUR_SIZE + 1) centred on that pixel.
//
//   BLUR_SIZE = 1  ->  3x3 window
//   BLUR_SIZE = 2  ->  5x5 window
//
// Rules:
//   1. One thread per output pixel, 2-D grid, 2-D blocks (same as grayscale).
//   2. Near the edges part of the window falls outside the image. Those
//      positions DO NOT count: sum only the in-bounds pixels and divide by
//      the number of in-bounds pixels you actually summed (not by the full
//      window size). Think about what happens at a corner.
//   3. Accumulate in an int, then divide, then store as unsigned char.
//   4. Do not read or write out of bounds (compute-sanitizer must be clean).
//
// Questions to answer in a comment at the bottom of this file once it works:
//   Q1. For a 3x3 window, how many global-memory LOADS does each thread do?
//       How many STOREs? What is the ratio of loads to stores, and why does
//       that make this kernel memory-bound?
//   Q2. Two neighbouring threads in the same block read mostly the same
//       pixels. Does the hardware do anything to help with that? (Hint:
//       think about L1/L2 cache; Chapter 5 will fix this properly with
//       shared memory.)
//   Q3. Why can't you just divide by (2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)
//       everywhere? Describe what the output would look like at the border
//       if you did.
//   Q4. Compare grayscale and blur: in grayscale every thread was
//       independent of its neighbours' pixels. What changed here, and why is
//       that the first real step toward "stencil" computations?

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include "cuda_check.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define BLUR_SIZE 3   // try 1, then 3, then 7 and compare out_blur.png

// ---------------------------------------------------------------------------
// YOUR PART 1: the kernel.
//   in  : w*h bytes, one unsigned char per pixel (grayscale)
//   out : w*h bytes
//   Loop over rows  y-BLUR_SIZE .. y+BLUR_SIZE
//        and cols   x-BLUR_SIZE .. x+BLUR_SIZE,
//   skip anything outside [0,w) x [0,h), count what you summed, divide.
// ---------------------------------------------------------------------------
__global__
void blurKernel(const unsigned char* in, unsigned char* out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < w && y < h) {
        int sum = 0;
        int count = 0;
        for(int k=-BLUR_SIZE; k <= BLUR_SIZE; k++) {            
            for(int l=-BLUR_SIZE; l <= BLUR_SIZE; l++) {
                int xi = x + k;                
                int yj = y + l;
                if(xi >= 0 && xi < w && yj >= 0 && yj < h) {
                    sum += in[yj * w + xi];
                    count++;
                }
            }
        }
        out[y * w + x] = sum / count;
    }
}

int main() {
    int w, h, c;
    unsigned char* in = stbi_load("out.png", &w, &h, &c, 1);  // force 1 channel: grayscale
    if (!in) { fprintf(stderr, "load failed: %s\n", stbi_failure_reason()); return 1; }

    if (w <= 0 || h <= 0) { fprintf(stderr, "bad dims\n"); stbi_image_free(in); return 1; }
    const size_t npix = (size_t)w * (size_t)h;
    if (npix > (size_t)INT_MAX) { fprintf(stderr, "image too large\n"); stbi_image_free(in); return 1; }

    unsigned char* out = (unsigned char*)malloc(npix);
    if (!out) { fprintf(stderr, "malloc failed\n"); stbi_image_free(in); return 1; }

    // -----------------------------------------------------------------------
    // YOUR PART 2: device side.
    //   cudaMalloc d_in / d_out (npix bytes each)
    unsigned char *d_in = nullptr;
    unsigned char *d_out = nullptr;
    int size = w * h * sizeof(unsigned char);

    CUDA_CHECK(cudaMalloc((void**) &d_in, size));
    CUDA_CHECK(cudaMalloc((void**) &d_out, size));
    //   copy `in` to d_in
    CUDA_CHECK(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));
    //   pick a 2-D block, compute a 2-D grid that covers w x h
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    //   launch blurKernel, check launch + execution errors
    blurKernel<<<grid, block>>>(d_in, d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
    //   copy d_out back into `out`
    CUDA_CHECK(cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost));

    //   free device memory
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    // -----------------------------------------------------------------------

    int ok = stbi_write_png("out_blur.png", w, h, 1, out, w);
    stbi_image_free(in);
    free(out);
    if (!ok) { fprintf(stderr, "write failed\n"); return 1; }
    return 0;
}

// Questions to answer in a comment at the bottom of this file once it works:
//   Q1. For a 3x3 window, how many global-memory LOADS does each thread do?
//       How many STOREs? What is the ratio of loads to stores, and why does
//       that make this kernel memory-bound?

/** 
** Answers to the questions:
for a 3x3 window, each thread does 9 global memory LOADS and 1 store.
why is that?
well thats because 3 rows and 3 collumns results in 9 distinct pixels 
which has 9 distinct memory locations, heace each pixel needs to be read
separately. which is what takes place inside the 2 for loops in the kernel.

the 3x3 window results in 1 store, because only one value is written to the output
which is the sum of the window.

**/

//   Q2. Two neighbouring threads in the same block read mostly the same
//       pixels. Does the hardware do anything to help with that? (Hint:
//       think about L1/L2 cache; Chapter 5 will fix this properly with
//       shared memory.)
/**
 * each threads has dedicated registers which store temp values, blocks share same L1 cache memory
 * so all threads in the same block benefit from same L1 cache memory.
 */

//   Q3. Why can't you just divide by (2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)
//       everywhere? Describe what the output would look like at the border
//       if you did.

/**
 * if you just divide by (2*BLUR_SIZE+1)*(2*BLUR_SIZE+1) everywhere, 
 * the border pixels would be averaged with non-existent pixels outside the image,
 *  leading to incorrect darkening or artifacts at the edges.
 */

//   Q4. Compare grayscale and blur: in grayscale every thread was
//       independent of its neighbours' pixels. What changed here, and why is
//       that the first real step toward "stencil" computations?

/*
* grayscale every thread is independedn and only reads it's own pixel
* blue on the other hand, each thread reads multiple neighboring pixels within the blur window
* this dependency on neighboring pixels is what makes it a stencil computation.
*/