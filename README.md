This repository is my theoretical and hands-on experience with GPU and CUDA. The main goal of this project is to understand how to optimize CUDA kernels by combining ideas from research papers, blog posts, and open-source samples.

The focus starts with General Matrix Multiplication (GEMM) — a core operation in both HPC and deep learning — because optimizing GEMM covers most of the key strategies needed for GPU programming. From there, I extended the ideas to other kernels like reduction, softmax, and matrix transpose.

The key philosophy I took away: high-performance GPU programming is all about managing data movement — from global memory → shared memory → registers — to keep compute units busy and hide latency.

My environment, on colab:

- CUDA Driver = CUDART

- CUDA Driver Version = 12.4

- CUDA Runtime Version = 12.5

- NumDevs = 1

| Property | Value |
|----------|-------|
| Device | Tesla T4 |
| CUDA Driver/Runtime Version | 12.4 / 12.5 |
| CUDA Capability | 7.5 |
| Global Memory | 15095 MBytes |
| Multiprocessors | 40 |
| CUDA Cores/MP | 64 |
| Total CUDA Cores | 2560 |
| GPU Max Clock | 1590 MHz |
| Memory Clock | 5001 MHz |
| Memory Bus Width | 256-bit |
| L2 Cache Size | 4194304 bytes |
| Shared Memory per Block | 49152 bytes |
| Registers per Block | 65536 |
| Warp Size | 32 |
| Max Threads per Block | 1024 |
| Max Block Dimensions | (1024, 1024, 64) |
| Max Grid Dimensions | (2147483647, 65535, 65535) |

# Part 1: Basic Knowledge of GPU
[Notes: CUDA Programming Course (High-Performance Computing with GPUs)](https://ymlinch.notion.site/CUDA-Programming-Course-High-Performance-Computing-with-GPUs-2786d2c70c38805e9215ef45b248a55c)

- **Execution model** — GPUs consist of Streaming Multiprocessors (SMs), each executing warps of 32 threads in lockstep. Threads are organized into blocks, and blocks form a grid. Divergence inside a warp reduces efficiency.
- **Memory hierarchy** — Registers (per-thread, fastest) → Shared memory/L1 (per-block) → L2 cache → Global device memory (large but high-latency). Performance depends heavily on minimizing global memory traffic and maximizing data reuse.
- **CUDA programming model** — Kernels are launched with a grid–block configuration (<<<grid, block>>>), mapping work to threads. Threads in a block can share data and synchronize with __syncthreads().
- **Streams and concurrency** — CUDA streams enable overlapping of computation and memory transfer, improving utilization.
- **Task graphs** — CUDA Graphs allow capturing kernel dependencies as a DAG for reduced launch overhead.
- **CUDA libraries** — cuBLAS, cuSPARSE, cuTENSOR, and Thrust provide highly optimized building blocks for scientific computing.
- **Distributed GPU computing** — At large scale, multiple GPUs across nodes communicate using MPI. Modern GPU-aware MPI can transfer data directly from device memory, avoiding unnecessary host transfers.


# Part 2: Deep Dive on GEMM Optimization

Started from the most basic matrix multiplication kernel and progressively optimized it step by step:
- Naive version — each thread computes one element of C, but suffers from non-coalesced global memory access, which kills bandwidth.
- Shared memory tiling (block tiling) — tiles of A and B are staged into shared memory, cutting down expensive global memory accesses.
- Register tiling (thread tiling) — each thread computes a sub-tile (e.g., 8×8) using registers, further improving reuse and arithmetic intensity.
- Double buffering (prefetching) — overlap memory loads and computation by alternating between two buffers, hiding shared memory latency.
- Optional assembly-level tuning — explored concepts like register bank conflicts, remapping, and instruction scheduling for the last few % of performance.

Through profiling and iteration, I saw how each optimization improved throughput and pushed the kernel closer to cuBLAS-level efficiency.

# Part 3: Extending to Other Kernels

- Reduction — replaced slow atomics with warp-level shuffle intrinsics for fast, lockstep reductions.
- Softmax — implemented a warp-per-row approach to combine max, sum, and normalization in a single pass, minimizing memory traffic.
- Matrix Transpose — used shared memory staging and padding/swizzling to avoid bank conflicts and enable coalesced reads/writes.
- Element-wise ops — experimented with float4 vectorized loads to maximize memory bandwidth.

# Sources

[1] How to Optimize a CUDA Matmul Kernel for cuBLAS-like ... https://siboehm.com/articles/22/CUDA-MMM

[2] GitHub - Tongkaio/CUDA_Kernel_Samples: CUDA 算子手撕与面试指南 https://github.com/Tongkaio/CUDA_Kernel_Samples

[3] https://www.youtube.com/watch?v=86FAWCzIe_4

[4] Kirk, D.B. and Wen-Mei, W.H., 2016. Programming massively parallel processors: a hands-on approach. Morgan kaufmann.

[5] https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/

[6] https://www.zhihu.com/column/c_1437330196193640448
