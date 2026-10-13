# Chapter 3: Matrix multiplication (naive)

`P = M x N` for square `WIDTH x WIDTH` row-major float matrices, one thread
per output element, no shared memory. This is the baseline that Chapter 5's
tiled version will be measured against.

**Status: in progress.** The host scaffolding (random fill, CPU reference,
tolerance-based verify, `cudaEvent` timing, GFLOP/s print) is in place.
The kernel body and the device-side allocation/launch code are still `TODO`.

## Build and run

```sh
nvcc -I common -o matmul ch03_matmul/matmul.cu && ./matmul; echo "exit=$?"
compute-sanitizer ./matmul
```

Expected output once complete:

```
WIDTH=512  kernel <ms> ms  <n> GFLOP/s
PASS
```

## Plan

- Thread `(x, y)` computes `P[row = y][col = x]`. 2-D grid of 2-D blocks,
  ceiling division, guard `row < width && col < width`.
- Loop `k = 0 .. width-1`, accumulate `M[row*width + k] * N[k*width + col]`
  in a register, store once.
- Build with `WIDTH = 500` at least once so the guard is exercised by a
  size that is not a multiple of the block dimension.
- Verify with a relative tolerance of `1e-4`, because the CPU and GPU sum
  the `WIDTH` products in different orders and the rounding differs.

## Questions to answer once it works

These are listed in the header of `matmul.cu`. In summary:

1. Loads, stores and FLOPs per thread, and the FLOP-per-byte ratio.
   Expected: `2*WIDTH` loads, 1 store, `2*WIDTH` FLOPs, so about 0.25
   FLOP/byte. Memory-bound by a wide margin.
2. How many times each element of `M` and `N` is loaded across the grid.
3. Which of the two access patterns (a row of `M` vs a column of `N` across
   32 consecutive threads) coalesces, and why.
4. Achieved GFLOP/s at `WIDTH` 512 and 1024 versus the GPU's FP32 peak.
5. Book exercise 3.1: one thread per row and one thread per column variants.

## Measurements

| WIDTH | kernel ms | GFLOP/s | % of FP32 peak |
|-------|-----------|---------|----------------|
| 512   | TBD       | TBD     | TBD            |
| 1024  | TBD       | TBD     | TBD            |

GPU: TBD. These numbers are the baseline Chapter 5 has to beat.
