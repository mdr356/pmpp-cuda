# Chapter 3: Box blur

Blur the grayscale image from the previous problem. Each output pixel is the
mean of the input pixels in a `(2*BLUR_SIZE+1)` square window centred on it.
This is the first kernel where a thread reads data that "belongs" to its
neighbours, which is the start of stencil computations.

## Build and run

Needs the stb headers in `ch03_grayscale/` and `out.png` produced by the
grayscale program in the working directory.

```sh
nvcc -I common -I ch03_grayscale -o blur ch03_blur/blur.cu
./blur               # reads out.png, writes out_blur.png
compute-sanitizer ./blur
```

`BLUR_SIZE` is a compile-time constant at the top of the file. 1 gives a 3x3
window, 3 gives 7x7, 7 gives 15x15.

## Design

**One thread per output pixel, 2-D grid**, same 16x16 blocks and ceiling
division grid as grayscale.

**Edge handling by counting, not by padding.** The window loop checks each
`(xi, yj)` against `[0, w) x [0, h)` and only accumulates in-bounds pixels,
keeping a running `count` of how many it summed. The final value is
`sum / count`, not `sum / window_area`. At a corner with `BLUR_SIZE = 1` only
4 of the 9 window positions exist, and the pixel is averaged over those 4.

**Integer accumulation.** `sum` is an `int`. The largest possible sum at
`BLUR_SIZE = 7` is `225 * 255 = 57375`, well within range, and avoids
floating-point rounding in the divide.

**No out-of-bounds access.** The in-bounds check happens before the load,
so `compute-sanitizer` is clean with any `BLUR_SIZE`.

## Questions from the exercise

**Q1. Loads and stores per thread.** For a 3x3 window, 9 global loads and 1
global store, a 9:1 ratio. The arithmetic is 9 adds and 1 divide, so the
kernel moves roughly one byte per operation. That is far below what any GPU
needs to be compute-bound, so this kernel is memory-bound, and the ratio
grows with the square of the window size.

**Q2. Neighbouring threads read mostly the same pixels.** Two horizontally
adjacent threads with a 3x3 window share 6 of their 9 loads. The hardware
helps implicitly: those loads hit in L1 (per-SM) or L2 (shared across the
chip) after the first thread brings the line in, so the DRAM traffic is much
lower than 9 bytes per thread. But it is still 9 load instructions per
thread, each going through the memory pipeline. Chapter 5 fixes this
properly by staging the tile into shared memory once and reading it from
there.

**Q3. Why not divide by the full window area everywhere.** Border pixels
would be averaged as if the missing out-of-image neighbours were zero. The
output would have a dark frame around the edge, one or more pixels wide
depending on `BLUR_SIZE`, with the corners darkest because they are missing
the most neighbours.

**Q4. What changed from grayscale.** In grayscale each thread read only its
own pixel; the threads were completely independent and the data access
pattern was one-to-one. Here each thread reads a neighbourhood, so the input
footprint of a block is larger than its output footprint and overlaps with
adjacent blocks. That overlap is the defining feature of a stencil, and
managing it (tiling, halos, shared memory) is what the next chapters are
about.

## Measurements

Not yet timed. The interesting comparison is kernel time at `BLUR_SIZE` 1, 3
and 7: the load count goes 9, 49, 225, and the time should scale closer to
that than to the fixed 1 store per thread.
