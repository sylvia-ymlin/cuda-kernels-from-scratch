This repository is my hands on experience on CUDA learning. 

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

This project serves as a comprehensive guide to writing high-performance code on NVIDIA GPUs. It synthesizes the principles and techniques from the provided articles and code samples to build a deep understanding of CUDA optimization. The primary focus will be on General Matrix Multiplication (GEMM), as it encompasses a wide range of crucial optimization strategies. Following this, the guide will cover other common and important kernels, including reduction, softmax, and matrix transpose, drawing heavily on the provided `CUDA_Kernel_Samples` repository [10].

The core philosophy of this guide is that high-performance computing is about managing data movement across the GPU's memory hierarchy (global memory -> shared memory -> registers) to hide memory latency and keep the compute units fed with data [provided articles on GEMM optimization].

### **Part 1: Deep Dive on GEMM Optimization**

GEMM is a foundational operation in HPC and the computational backbone of modern deep learning models, making its optimization a critical skill [provided articles on GEMM optimization]. This section follows a step-by-step refinement process, from a simple implementation to a highly optimized kernel.

### **1. The Naive Approach and Global Memory Issues**

The most straightforward implementation assigns one CUDA thread to compute a single element of the output matrix C. The thread calculates the dot product of a row from matrix A and a column from matrix B [10].

This kernel, while functionally correct, performs poorly due to **non-coalesced global memory access**. When threads in a warp access elements from a column of matrix B (stored in row-major order), their memory requests are scattered, leading to many slow, individual memory transactions. This severely underutilizes the GPU's memory bandwidth [provided articles on GEMM optimization].

### **2. From Global to Shared Memory (Block Tiling)**

To address the global memory bottleneck, we introduce **block tiling**. The core idea is to reduce the total number of accesses to slow global memory by exploiting data reuse [provided articles on GEMM optimization].

- **Concept**: The input matrices A and B are partitioned into smaller tiles. Each thread block is responsible for computing one tile of the output matrix C. It does this by iterating through the necessary tiles of A and B, loading them into the fast on-chip **shared memory** first.
- **Benefit**: Once a tile is in shared memory, all threads in the block can access it repeatedly with much lower latency than going to global memory. This dramatically reduces global memory traffic and improves performance [10][provided articles on GEMM optimization].

### **3. From Shared Memory to Registers (Thread Tiling)**

Even with shared memory, access to it can become the next bottleneck. To further increase arithmetic intensity (the ratio of math operations to memory operations), we bring the data even closer to the execution units using **thread tiling** [provided articles on GEMM optimization].

- **Concept**: Instead of computing a single output element, each thread is now responsible for a small 2D sub-tile (e.g., 8x8) of the block's output tile.
- **Benefit**: This allows the thread to load values from shared memory into its private **registers**—the fastest form of memory—and reuse them for multiple calculations. This reduces shared memory traffic and allows the compute units to be more effectively utilized [10][provided articles on GEMM optimization].

### **4. Hiding Latency with Data Prefetching (Double Buffering)**

Memory access, even to shared memory, has latency. **Data prefetching**, also known as double buffering, is a pipeline technique used to hide this latency.

- **Concept**: We allocate two buffers in shared memory and registers. While the compute units are processing data from the `read` buffer (for the current iteration), the memory units are simultaneously fetching data for the *next* iteration into the `write` buffer. In the next iteration, the buffers swap roles.
- **Benefit**: This overlapping of computation and memory transfer ensures that the compute units do not have to stall and wait for data, effectively hiding memory latency and keeping the GPU "fed" [provided articles on GEMM optimization].

### **5. Advanced Tuning: Assembly-Level Insights (Optional)**

The final frontier of optimization involves analyzing and sometimes manually tuning the assembly code (SASS) generated by the compiler. While significant performance (e.g., 91-97% of cuBLAS) is achievable with the CUDA C techniques above, this step is for extracting the last few percentage points of performance [provided articles on GEMM optimization].

- **Register Bank Conflicts**: NVIDIA GPUs have register files divided into banks. If a single instruction needs multiple source operands that reside in the same bank, a conflict occurs, and the instruction is re-issued, wasting a cycle.
- **Mitigation**: This can be solved with **register remapping** (carefully choosing which registers hold which values) and **instruction rescheduling** to break dependencies and better utilize features like the reuse cache [provided articles on GEMM optimization].

### **Part 2: Optimizing Other Common Kernels**

The principles from the GEMM optimization apply broadly. This section covers other kernels from the `CUDA_Kernel_Samples` repository [10].

### **1. Reduction (High Importance)**

Reduction kernels (e.g., sum, max) aggregate an array into a single value.

- **Naive Method**: Using `atomicAdd` in a loop serializes execution and destroys parallelism [10].
- **Recommended Method (Warp-Level Reduction)**: The most efficient approach uses warp-level shuffle intrinsics like `__shfl_down_sync()`. Since all 32 threads in a warp execute in lockstep, they can exchange data and perform a parallel reduction without needing a slow, block-wide `__syncthreads()` synchronization. The final result from each warp is then aggregated [10].

### **2. Softmax (High Importance)**

Softmax is common in deep learning and requires finding a max value and a sum to normalize a vector.

- **Inefficient Method**: Launching separate kernels for finding the max, calculating the sum, and normalizing is simple but slow due to kernel launch overhead [10].
- **Recommended Method (Warp-per-Row)**: For a matrix, a single kernel can assign one warp to each row. The warp uses the efficient warp-level reduction techniques to find its row's max and sum. All calculations are done in one pass, minimizing memory traffic and overhead [10].

### **3. Matrix Transpose (Medium Importance)**

A naive transpose results in non-coalesced reads and writes.

- **Recommended Method (Shared Memory Staging)**: Use shared memory as a staging area. Threads perform a **coalesced read** of a tile from the input matrix into shared memory. After a `__syncthreads()`, they perform a **coalesced write** of the now-transposed tile to the output matrix. This avoids shared memory **bank conflicts** by either padding the shared memory array or using an address swizzling technique (e.g., `threadIdx.x ^ threadIdx.y`) [10].

### **4. Element-wise Kernels (Low Importance)**

These kernels (e.g., vector add, ReLU) are memory-bound. The primary optimization is **vectorized memory access**. Instead of operating on one `float` at a time, the kernel should use `float4` to load, process, and store four elements in a single instruction, maximizing memory bandwidth [10][provided articles on GEMM optimization].

# Sources

[1] How to Optimize a CUDA Matmul Kernel for cuBLAS-like ... https://siboehm.com/articles/22/CUDA-MMM

[2] 6 Step Optimization of GeMMs in CUDA - Rimika Writes https://www.rimikawrites.com/6-step-optimization-of-gemms-in-cuda/

[3] Performance Analysis of CUDA-based General Matrix ... http://kth.diva-portal.org/smash/get/diva2:1985710/FULLTEXT01.pdf

[4] Advanced NVIDIA CUDA Kernel Optimization Techniques https://developer.nvidia.com/blog/advanced-nvidia-cuda-kernel-optimization-techniques-handwritten-ptx/

[5] Understanding GEMM Performance and Energy on NVIDIA ... https://arxiv.org/html/2411.16954v1

[6] Advanced Matrix Multiplication Optimization on NVIDIA GPUs https://salykova.github.io/sgemm-gpu

[7] GPU optimization techniques to accelerate optiGAN—a ... https://pmc.ncbi.nlm.nih.gov/articles/PMC11170465/

[8] A methodology for comparing optimization algorithms ... https://www.sciencedirect.com/science/article/pii/S0167739X24002498

[9] Profiling and optimization of multi-card GPU machine ... https://arxiv.org/html/2505.22905v1

[10] GitHub - Tongkaio/CUDA_Kernel_Samples: CUDA 算子手撕与面试指南 https://github.com/Tongkaio/CUDA_Kernel_Samples

[11] https://www.youtube.com/watch?v=86FAWCzIe_4

[12] Kirk, D.B. and Wen-Mei, W.H., 2016. Programming massively parallel processors: a hands-on approach. Morgan kaufmann.
