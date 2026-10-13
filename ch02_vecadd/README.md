# Chapter 2: Vector add

`C[i] = A[i] + B[i]` for `N` floats. The "hello world" of CUDA, but it is the
template every later problem follows: allocate on the host, allocate on the
device, copy in, launch, copy out, verify, free.

## Build and run

```sh
nvcc -I common -o vecadd ch02_vecadd/vecadd.cu && ./vecadd; echo "exit=$?"
compute-sanitizer ./vecadd
```

Prints `Verification passed!` and exits 0, or prints the first mismatching
index and exits 1.

## Design

**One thread per element, 1-D grid.** Global index is
`blockIdx.x * blockDim.x + threadIdx.x`. Block size is 256 threads, grid size
is `ceil(N / 256)`.

**Bounds guard.** `N = 1000` is deliberately not a multiple of 256, so the
grid has 4 blocks = 1024 threads and the last 24 must do nothing. Without
`if (i < N)` they would write 96 bytes past the end of `C`, which
`compute-sanitizer` catches immediately.

**Verification on the host.** The CPU recomputes every `A[i] + B[i]` and
compares exactly. Exact comparison is fine here because each output is a
single addition with no reordering, so CPU and GPU produce identical bits.
Matmul will need a tolerance for exactly that reason.

**Error handling.** Every runtime call goes through `CUDA_CHECK`, and the
launch is followed by `cudaGetLastError()`. The `cudaMemcpy` back to the host
is synchronous, so it also surfaces any execution error from the kernel.

## What this kernel is bound by

Each thread does 2 loads and 1 store of 4 bytes each for one floating-point
add: 12 bytes moved per FLOP. No GPU can feed that from DRAM at anything near
its compute peak, so this kernel is purely memory-bandwidth bound. At `N =
1000` it is actually launch-latency bound; the kernel finishes long before
the overhead of launching it is paid for. That is expected and not worth
optimizing. The point of the exercise is the host-side discipline.

## Measurements

Not yet timed. When run on a GPU, the useful number is effective bandwidth
(`3 * N * 4 bytes / kernel time`) at a large `N` such as 2^26, compared to
the device's published memory bandwidth.
