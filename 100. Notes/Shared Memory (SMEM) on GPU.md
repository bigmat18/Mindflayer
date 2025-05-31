**Data time:** 16:16 - 31-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Shared Memory (SMEM) on GPU

It is essentially a **scratch-pad memory** directly managed by the programmer (unlike caches) and allocated within a SM. Once a buffer is allocated in SMEM, it is accessed by all threads in the same block. It composed of **4-bytes words (32 bits)**. If a thread reads a byte, the whole word is read from SMEM. SMEM consists of **32 banks**, each one able to provide a word in 2 clock cycles (one for the request, one for getting the word).

We can allocate SMEM in two possible ways:
- **Statically** (`__shared__`) if the size is known at compile time
- **Dynamically** (`extern __shared__`) otherwise

On **Ampere GPU** for example, we have **100 KiB** of SMEM per SM, where a block can use up to **99 KiB** (1 KiB is reserved). The usage of SMEM by a block limits the number of resident blocks in the same SM. SMEM is of great importance to reduce the memory latency.

If a kernel uses a shared-memory buffer of a statically known size, this can be simply declared in the kernel code:
![[Pasted image 20250531162059.png]]

If the size of the shared-memory buffer is not known at compile time but only at **runtime**:
![[Pasted image 20250531162123.png]]

In both cases, the array `smem_buf` is available **one per block** of the grid launched with the kernel `kernel_foo`. Shared memory is on-chip, much faster than global memory.

### SMEM Organization
Each SMEM module (one per SM) is composed of **32 banks**. Each bank can be used to **read/write a 32-bit word** with one request served in **2 clock cycles**. Words are distributed among banks in an **interleaved manner**. Threads of each warp generate in parallel a request to SMEM.

![[Pasted image 20250531162419.png | 500]]

- **Bast case**: threads read a word each, each bank is accessed by one thread (hardware **generates** one 128-byte transaction).
- **Worst case**: all threads of a warp access the same bank for a different word (we need 32 4-byte transactions)

### SMEM Bank Conflicts
###### No Conflict
if a LOAD/STORE instruction requests at most one access per bank, we do not have conflict. Even if accesses by threads of the warp are not aligned, if we have one access per bank no conflict arises (all words can be retrieved in parallel in 2 clock cyles)

![[Pasted image 20250531162626.png]]
###### Conflict
more threads of a warp access the same bank for different addresses. In the figure below, only 8 words can be accessed simoultaneously, we pay 4 accesses sequentially.

![[Pasted image 20250531162634.png]]

### SMEM Access Patters
##### Broadcast Read
A warp accesses only one bank for the same word.
![[Pasted image 20250531162916.png | 650]]
##### Parallel Read
Each thread in the block reads a different word in a different bank.
![[Pasted image 20250531162955.png | 650]]
##### Conflict Read
Each thread accesses an element of the array (4 bytes) at position double of the thread identifier.
![[Pasted image 20250531163044.png | 650]]
##### No-Conflict Read
Each thread accesses an element of the array (4 bytes) at position three times of the thread identifier.
![[Pasted image 20250531163118.png |650]]

### Multiple SMEM Allocation
The delay to receive data from the SMEM can be masked by the **warp scheduler**, since that warp is marked idle and others might be run on the stream multi-processor. SMEM bank conflicts may happen only between threads of the same warp. To allocate more arrays in the same SMEM allocated area, we have to properly use offsets to the specific arrays.
```c
__global__ void kernel(…)
{
	extern __shared__ float array[];
	float *array0 = array;
	int *array1 = (int *) array + 64;
	…
}

int main(void)
{
	int nBytes = 64*sizeof(float)+256*sizeof(int);
	kernel_name<<<…, …, nBytes>>>();
	cudaDeviceSynchronize();
	return 0;
}
```
# References