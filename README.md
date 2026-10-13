# pmpp-cuda

Working through *Programming Massively Parallel Processors* (PMPP, 4th ed.) in CUDA,
one problem per directory. Every kernel is written from scratch, checked with
`compute-sanitizer`, and verified against a CPU reference where one makes sense.

The goal is not to collect kernels. It is to be able to explain, for each one,
where the threads go, where the bytes go, and why the kernel is as fast or as
slow as it is.

## Problems

| Chapter | Problem | What it teaches | Status |
|---------|---------|-----------------|--------|
| 2 | [Vector add](ch02_vecadd/) | 1-D grid, host/device memory round trip, bounds guard | Done, verified |
| 3 | [Grayscale](ch03_grayscale/) | 2-D grid over an image, interleaved RGB indexing | Done |
| 3 | [Box blur](ch03_blur/) | Stencil access, edge handling, load/store ratio | Done, questions answered |
| 3 | [Matrix multiply](ch03_matmul/) | One thread per output, compute-to-memory ratio, GFLOP/s | In progress |
| 4 | [Warps and occupancy](ch04_warps/) | Device limits, control divergence cost, block size vs occupancy | Not started |

Each directory has its own README with the problem statement, the design
decisions, the build command, and answers to the chapter's questions.

## Layout

```
common/cuda_check.cuh   CUDA_CHECK(): abort with name + message on any runtime error
ch02_vecadd/vecadd.cu
ch03_grayscale/grayscale.cu
ch03_blur/blur.cu
ch03_matmul/matmul.cu
ch04_warps/warps.cu
```

## Building

Everything is a single `.cu` file. Build from the repo root with `nvcc`:

```sh
nvcc -I common -o vecadd ch02_vecadd/vecadd.cu && ./vecadd
```

The image problems also need the single-header
[stb_image](https://github.com/nothings/stb) loaders. Drop `stb_image.h` and
`stb_image_write.h` into `ch03_grayscale/` and add `-I ch03_grayscale` to the
build line. They are not checked in.

Every binary is run under the sanitizer before it is considered done:

```sh
compute-sanitizer ./vecadd
```

## Conventions

- **Every CUDA call is wrapped in `CUDA_CHECK`.** Kernel launches are followed
  by `cudaGetLastError()` for launch errors and, where it matters,
  `cudaDeviceSynchronize()` so execution errors are attributed to the right
  kernel.
- **Grids are sized with ceiling division** so a problem size that is not a
  multiple of the block size still gets covered, and every kernel guards
  `if (i < N)` so the overhang threads do nothing.
- **Verification before timing.** A kernel that prints `PASS` against a CPU
  reference gets timed. One that does not gets fixed.

## Measurements

Timings live in each problem's README once the kernel has been run on an
NVIDIA GPU. This code is developed on a Mac with no CUDA device, so a problem
without a timing table has simply not been measured yet.
