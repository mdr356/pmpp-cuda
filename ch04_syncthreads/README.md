# Chapter 4: `__syncthreads()` and block-level barriers

Section 4.3 introduces the one synchronization primitive the book uses for
the rest of the text: `__syncthreads()`, a barrier that holds every thread in
a block until all of them arrive. It is what makes shared memory usable as a
hand-off between threads, and it is also the reason a block must be resident
on one SM in its entirety, which section 4.7 turns into the occupancy limits.
Chapter 5's tiled matrix multiply is built on it, so this problem isolates
the primitive before it gets buried in a bigger kernel.

**Status: not started.** Scaffolding is in `syncthreads.cu`; the three
`TODO` kernels and two `TODO` host sections are the work.

## Build and run

```sh
nvcc -I common -o syncthreads ch04_syncthreads/syncthreads.cu && ./syncthreads; echo "exit=$?"
compute-sanitizer ./syncthreads
compute-sanitizer --tool racecheck ./syncthreads
```

The third line is new for this repo. `racecheck` watches shared memory and
reports any read that could have seen a write from another thread without a
barrier in between. It turns "the no-barrier kernel happened to pass" into a
deterministic failure.

## The two experiments

**Part A: block reverse through shared memory.** Each block loads its slice
of the input into a shared tile, hits the barrier, then every thread stores
the element from the *opposite* end of the tile. The write and the read are
done by different threads, so nothing but the barrier orders them. The
kernel is written twice, once with the barrier and once without. The
no-barrier version is expected to be wrong, and the interesting part is
that it may still pass on some runs, which is why the sanitizer is part of
the definition of done.

`N` is deliberately not a multiple of the block size. The last block has
threads with nothing to load or store, but those threads must still reach
the barrier, so the bounds guards wrap the memory accesses and not the
`__syncthreads()`. Getting that distinction wrong is the classic barrier
bug and Q1 is about it.

**Part B: barrier cost.** One kernel does a fixed amount of arithmetic with
0, 1, 16, 64 or 256 barriers interleaved. The arithmetic is identical across
launches, so the slope of time against barrier count is the cost of a
barrier. Run at block sizes 128 and 1024 to see whether a barrier across
32 warps costs more than one across 4.

## Questions

Five questions are listed at the top of `syncthreads.cu`. They cover the
barrier-inside-a-branch rule, reading a `racecheck` report, the measured
barrier cost in cycles, why there is no grid-wide barrier (transparent
scalability), and what each of the two barriers in Chapter 5's tiled matmul
protects.

## Measurements

| Experiment | Result |
|------------|--------|
| `reverseNoBarrierKernel` verify result (3 runs) | TBD |
| `racecheck` hazards: with barrier / without | TBD / TBD |
| Barrier cost, 128-thread block | TBD ns (TBD cycles) |
| Barrier cost, 1024-thread block | TBD ns (TBD cycles) |

GPU: TBD.
