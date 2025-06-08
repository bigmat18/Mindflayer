**Data time:** 15:11 - 31-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# GPU Memory Hierarchy

The GPU memory hirarchy consists of **programmable memories** (i.e., users can decide which data should be allocated/deallocated explicitly), or **non-programmable memories** like L2/L1 caches. Memory physical supports:
- **Device Memory**: off-chip, $10¹ - 10²$ GiBs of capacity bandwitdh > 1 TiB/s with HBM
- **L2 Cache**: on-chip, $10¹ - 10^2$  MiBs, bandwidth 3-5 TiB/s
- **L1/Shared Memory**: $10⁰-10¹$ MiBs per SM, bandwidth $10⁰ -10¹$ of TiB/s per SN
- **Registers**: order of $10^1 − 10^2$ 32-bit registers per SM, access time of 1 clock cycle

Going down in the hierarchy, memories are smaller and fasters (as usual). Modern GPUs provide a very high memory bandwith (even reaching order of TiB/s) with **3D stacked memories (HBM)** for the off-chip device memory.

Each **SM** is equipped with some **CUDA cores**, a **register file**, a **control logic** to run the resident warps of resident blocks. Furthermore, the SM is equipped with an **L1 cache**, and **SMEM**. Different SMs share a L2 and the **off-chip device memory**.

![[Pasted image 20250531152707.png | 500]]

### Logical CUDA Memories 

![[Pasted image 20250531154123.png | 200]]
###### Global Memory
- Allocated in the **off-chip device memory**. Cached on **L1/L2**
- Accessible by all threads of different kernels launched by the same CUDA context
###### Texture Memory
- Allocated in the **off-chip device memory**
- Cached on **L1/L2** in read-only mode and in an optimized manner (FP interpolation and filtering)
- Accessible by all threads of different kernels launched by the same CUDA context
###### Constant Memory
- Bounded to 64 KiB, it is allocated in the **off-chip device** memory and cached in **L2 and L1**
- Efficient if more threads read the same location of constant memory (**broadcasting**)
- Accessible by all threads of different kernels launched by the same CUDA context
###### Local Memory
- Allocated in the **off-chip device memory** and cached in **L1/L2**
- Accessibly by one thread only of a grid (its lifetime is the one of the kernel)
###### Shared Memory
- Allocated in the **Shared Memory** of the SM, accessible by all threads of a block. It exists on a per-block basis

![[Pasted image 20250531154156.png | 500]]

![[Pasted image 20250531154233.png | 500]]

### GPU Caches
As for CPUs, caches are **non programmable**. Different kinds of caches in NVIDIA GPU devices
- **L1** one per stream multi-processor (used for caching **global** and **local memories**)
- **L2** shared by all stream multi-processors (used for caching **global** and **local memories**)
- **Read-only texture cache** one per stream multi-processors to **cache constant** and **texture memory**

![[Pasted image 20250531154450.png|600]]

![[Pasted image 20250531154541.png|600]]

### Registers
Fastest memory, one **Register File** for each stream multi-processor (SM). Lifetime: **kernel scope**. They are partitioned among the **resident warps** on that SM.  **Automatic variables** in device code are usually allocated on registers. The more registers a block uses, the fewer blocks can be resident in the same SM at the same time.

**Register spinning**: if we use more registers than what is available, the compiler tries to use the local memory.

Below an example of some local variables used by a thread in a kernel:
![[Pasted image 20250531155140.png | 250]]
- 4 variables of type int or `unsigned int`, for a total of 16 bytes per thread
- Blocks of 128 threads need 512 registers
- Blocks of 512 threads need 2K registers
- Blocks of 1024 threads need 4K registers
### Local Memory
It is allocated in the **off-chip device memory**. So, it is slow as the global memory. It is a memory local to a specic thread in a block. It is used to allocate **thread-private local arrays**, large data structures, or for storing registers in case of register spilling. From CC 2.x, local memory is cached in L2 and L1.
```c
__global__ void kernel_name(…)
{
	int local_array[100];
	…
}
```

In the example above, the compiler does not allocate 100 registers for each `local_array` for each thread, but rather such arrays are allocated in local memory. You can inspect the decisions taken by the compiler with specific options like `-–ptxas-options=-`v to understand the allocation choices.

### Constant Memory
Still allocated on the **off-chip device memory**. A constant is declared with the qualifier `__constant__`. A constant variable must be declared in the **global scope**, so outside the scope of any kernel. It can be accessed by all kernels of the same CUDA context.
```c++
__constant__ int A;
int main()
{
	float pA=1;
	...
	cudaMemcpyToSymbol(&A, &pA, sizeof(A));
	...
}
__device__ kernel_name(…)
{
	…
	// you can use A here
}
```
Initialization of constant memory can be done in the host program through the `cudaMemcpyToSymbol` primitive instead of using standard `cudaMemcpy`.

### Texture Memory
It is allocated in the **off-chip device memory** and can be accessed only through a **dedicated read-only L1 cache** per stream multi-processor. Such cache provides efficient hardware support for **floating-**
**point interpolation** and **filtering**.

**Example** of a computation requiring floating-point interpolation is a rotation of an image of, for example, 30 degrees (in general any angle not multiple of 90).

![[Pasted image 20250531160633.png | 350]]

The **read-only texture cache** is optimized for locality in the coordinate space of the texture (and not in terms of memory addresses as usual). The API for creating textures is a bit complicated. We will not study textures during the course.

### Global Memory
It is the largest, with the highest latency, and allocated in the off-chip device memory. Global derives from the fact that such a memory can be used by any kernel launched by the same CUDA context. It can be declared:
- **Statically**: `__device__ int array[10];`
- **Dynamically** `cudaMalloc(&array, sizeof(int)*10);`

All threads of the kernel, or of different kernels, might access the same locations in global memory (this might generate **race conditions**). Accesses to the global memory happen with transactions of 32, 64 or 128 bytes.

The accesses to the global memory have a speed that depends on:
- Which addresses the threads in the same warp generates
- How is aligned the global memory buffer
# References