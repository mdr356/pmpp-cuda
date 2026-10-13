# Chapter 4: Warps, divergence, and occupancy

Chapter 4 is about how the hardware schedules the kernels from Chapters 2
and 3: blocks are split into 32-thread warps, warps execute in lockstep, a
branch that splits a warp serialises both sides, and the number of resident
warps per SM is what hides memory latency. None of that is visible in the
earlier code, so this problem makes it measurable.

**Status: not started.** Scaffolding is in `warps.cu`; the three `TODO`
sections are the work.

## Build and run

```sh
nvcc -I common -o warps ch04_warps/warps.cu && ./warps; echo "exit=$?"
compute-sanitizer ./warps
```

## The three experiments

**Part A: device query.** Print the SM count, warp size, and the per-SM
limits on threads, blocks, registers and shared memory. Everything in the
rest of the chapter is a consequence of these numbers.

**Part B: control divergence.** Two kernels do identical total work on 16M
floats: half the threads run one arithmetic path, half run another. In the
divergent kernel the split is by `i % 2`, so every warp contains both paths
and the scheduler must run them one after the other. In the uniform kernel
the split is by `(i / 32) % 2`, so each warp takes a single path. Same
instructions, same data, different grouping. The timing difference is the
cost of divergence.

**Part C: block-size sweep.** A trivial memory-bound kernel launched with
block sizes from 32 to 1024. For each, time it and ask the runtime (via
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) how many blocks fit per SM,
then convert that to theoretical occupancy. The question is whether
occupancy predicts the timing, and where it stops mattering.

## Questions

Five questions are listed at the top of `warps.cu`. They connect the
measurements back to the chapter: resident warps per SM, SIMD efficiency of
the `vecadd` guard warp, why divergence costs less than 2x, and why a
memory-bound kernel saturates bandwidth before 100% occupancy.

## Measurements

| Experiment | Result |
|------------|--------|
| Divergent / uniform time ratio | TBD |
| Fastest block size (Part C) | TBD |
| Occupancy at that block size | TBD |

GPU: TBD.
