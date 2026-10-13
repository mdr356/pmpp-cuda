# Chapter 3: RGB to grayscale

Load a colour image, convert every pixel to one luminance byte on the GPU,
write it back out as a PNG. First use of a 2-D grid.

## Build and run

Needs `stb_image.h` and `stb_image_write.h` from
[nothings/stb](https://github.com/nothings/stb) in this directory. They are
not checked in.

```sh
nvcc -I common -I ch03_grayscale -o grayscale ch03_grayscale/grayscale.cu
./grayscale          # reads in.jpg, writes out.png
compute-sanitizer ./grayscale
```

`out.png` is the input for the [blur](../ch03_blur/) problem.

## Design

**One thread per pixel, 2-D grid.** Blocks are 16x16 = 256 threads. The grid
is `ceil(w/16) x ceil(h/16)`, and the kernel guards `x < w && y < h` so an
image whose dimensions are not multiples of 16 is still safe.

**Two index spaces.** The output is one byte per pixel at `y * w + x`. The
input is interleaved RGB, so the same pixel's red channel is at
`3 * (y * w + x)` with green and blue at `+1` and `+2`. Getting this mapping
right is the whole exercise. `stbi_load` is called with `req_comp = 3` so the
layout is guaranteed regardless of whether the source file has an alpha
channel.

**Luminance formula.** `0.299 R + 0.587 G + 0.114 B` (ITU-R BT.601), with
`+ 0.5f` before the truncating cast so the result rounds to nearest instead
of always down.

**Error attribution.** The launch is followed by both `cudaGetLastError()`
(bad launch configuration) and `cudaDeviceSynchronize()` (faults during
execution), so a failure is reported against this kernel rather than
surfacing later in an unrelated `cudaMemcpy`.

**Host-side guards.** Dimensions are checked for overflow before computing
`w * h * 3`, and every failure path frees what it allocated.

## What this kernel is bound by

3 byte loads and 1 byte store per thread for 3 multiplies and 2 adds. Memory
bound, like vector add, but with a better ratio because the loads are bytes
rather than floats. Every thread touches only its own pixel, so there is no
reuse between threads to exploit. The blur problem changes that.

## Measurements

Not yet timed.
