#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__
void grayKernel(const unsigned char* rgb, unsigned char* gray, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int idx = y * w + x;
        int rgbIdx = idx * 3;
        gray[idx] = static_cast<unsigned char>(
            0.299f * rgb[rgbIdx] +
            0.587f * rgb[rgbIdx + 1] +
            0.114f * rgb[rgbIdx + 2] + 0.5f
        );
    }
}

// equation for grayscale: gray[i] = 0.299f * R[i] + 0.587f * G[i] + 0.114f * B[i]

int main() {
    int w, h, c;
    unsigned char* rgb = stbi_load("in.jpg", &w, &h, &c, 3);  // force 3 channels, RGB interleaved
    if (!rgb) { fprintf(stderr, "load failed: %s\n", stbi_failure_reason()); return 1; }

    if (w <= 0 || h <= 0) { fprintf(stderr, "bad dims\n"); stbi_image_free(rgb); return 1; }
    const size_t npix = (size_t)w * (size_t)h;
    if (npix > (size_t)INT_MAX / 3) { fprintf(stderr, "image too large\n"); stbi_image_free(rgb); return 1; }
    const size_t rgbBytes = npix * 3;

    unsigned char* gray = (unsigned char*)malloc(npix);
    if (!gray) { fprintf(stderr, "malloc failed\n"); stbi_image_free(rgb); return 1; }

    // your part: cudaMalloc, copy rgb (w*h*3 bytes) to device, launch kernel,
    unsigned char *d_rgb = nullptr;
    unsigned char *d_gray = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgbBytes));
    CUDA_CHECK(cudaMalloc(&d_gray, npix));
    CUDA_CHECK(cudaMemcpy(d_rgb, rgb, rgbBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1)/ block.y);


    // launch kernel (assuming kernel is defined elsewhere)
    grayKernel<<<grid, block>>>(d_rgb, d_gray, w, h);
    CUDA_CHECK(cudaGetLastError());        // launch errors
    CUDA_CHECK(cudaDeviceSynchronize());   // execution errors, attributed to the kernel


    CUDA_CHECK(cudaMemcpy(gray, d_gray, npix, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    // copy gray (w*h bytes) back

    int ok = stbi_write_png("out.png", w, h, 1, gray, w);
    stbi_image_free(rgb);
    free(gray);
    if (!ok) { fprintf(stderr, "write failed\n"); return 1; }
    
    return 0;
}