**Data time:** 12:18 - 30-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Execution Model]]

**Area**: [[Master's degree]]
# CUDA and GPU hardware

A GPU consists of multiple **Streaming Multiprocessor (SMs)**, each consisting of multiple **cores** with shared **control** and **memory**

![[Pasted image 20250530125434.png | 400]]

Each **[[CUDA Kernels|kernel]]** is a grid of **blocks**, where each block contains several **threads**. Each GPU is a set of **Stream- Multiprocessors**, each SM contains several CUDA cores. Mapping:
- Each thread has its **local memory** for the stack and it is run by a CUDA core
- Each block of threads of the grid is executed by a SM
- Threads of the same block can **synchronize** (i.e., [[Barriers|barriers]])
- Threads of the same block can use a high-bandwidth low-latency **[[Shared Memory Architectures|shared memory]]**
- More kernels can access the same global memory

![[Pasted image 20250530125718.png | 250]]

### Thread Scheduling
Threads are assigned to SMs at block granularity (all threads in a block are assigned to the same SM). Threads/blocks require resources to execute (e.g., registers, memory, …) so SMs can accommodate a limited number of threads/blocks at once… The remaining blocks wait for other blocks to finish before they can be assigned to an SM.

![[Pasted image 20250530130129.png|400]]

Threads assigned to an SM run in parallel or concurrently. The SM has a scheduler that manages their execution. Blocks assigned to an SM are further divided into **warps** which are the **unit of scheduling**

![[Pasted image 20250530130310.png | 400]]

Threads in a warp are scheduled on 32 cores of an SM, which executes them following the **[[SIMD (Single Instruction, Multiple Data)|SIMD model]]**. Each instruction of the warp threads is run in parallel on the same number of cores, each working on a different data item. Blocks assigned to an SM are further divided into **warps** which are the unit of scheduling.

![[Pasted image 20250530131046.png | 150]]

### Scheduling Considerations
Threads in the same block are assigned to the same SM. Assigning threads in the same block to the same SM makes supporting collaboration between them efficient.

![[Pasted image 20250531115335.png | 450]]

All threads in a block are assigned to an SM simultaneously. A block cannot be assigned to an SM until it secures enough resources for all its threads to execute.

![[Pasted image 20250531115433.png | 500]]

Otherwise, if some threads reach a barrier and others cannot execute, the system could deadlock.

### [[Control Divergence on NVIDIA]]

### [[SIMT and Synchronization]]

### [[GPU Registers]]

### [[Logical & Physical View of Threads-Warps]]

### [[Atomic Instructions]]
# References