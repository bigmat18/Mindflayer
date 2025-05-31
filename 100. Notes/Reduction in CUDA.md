**Data time:** 21:55 - 31-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Reduction in CUDA

[[Reduce]] is a computation applied over an array A of elements $t$. The results is a single element $r$ of type $t$ such that:
$$
r= A[0] \oplus \dots \oplus A[L-1]
$$
where L is the size of the array. The sequential version is the following:
```c
int reduction(int *A, int L)
{
	int r = 0;
	for (int i=0; i<L; i++)
		r += A[i];
	return r;
}
```

**Linear complexity** in the size of the array. The basic algorithm is strictly sequential, iterations cannot be run in parallel (check [[Bernstein Conditions]]) In the following, we assume as the binary operator the **sum**, and T is **integer**.

By violating the owner-compute-rule, we can image L **virtual processors**, each reading one element of the array and writing the same output result variable (in an **atomic manner**)

### Naive Solution
The naïve solution is inspired by such a principle, with 𝑳 CUDA threads, one per VP of our logical model. This can be done with **atomic RMW instructions**. The idea is to read the elements of the input array in parallel by different CUDA threads, while the update of the final result variable is done with atomics.

**Example** with a **1D grid** of **1D blocks**
```c
__global__ void naive_reduction(int *A, int L, int *sum)
{
	unsigned int tUID = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tUID >= L) {
		return;
	}
	else {
		atomicAdd(sum, A[tUID]);
	}
}
```

The variable sum is allocated in global memory and initialized to zero before launching the kernel. Only memory reading is parallelized. Limited scalability.

The number of RMW instructions can be reduced to one per block or one per warp using the idea shown in the previous class about cooperative groups.

### Blocked Solution
A parallel approach can be enforced by leveraging the **associative property** of the binary operator. The **Approach** is based on every thread adds two elements in each step. Takes $\log(L)$ steps and half the threads drop out every step.

![[Pasted image 20250531220612.png | 400]]

In the following part, we assume $L = 2^x$ for some integer $x$. Threads proceed in multiple steps. Before passing from one step to the next one, a **[[Barriers]]** is enforced. Since it is not possible to synchronize all threads of a grid (except with cooperative groups that is a CUDA extension), we restrict the kernel to have **one 1D block only**.

![[Pasted image 20250531220747.png]]

- We launch a kernel with 1 block of $L/2$ threads (in the left-hand side figure, we have $L=16$ elements and 8 threads).
- Since a block can have 1024 threads at most, this kernel work with arrays of size $L \leq 2048$
- At each step, the number of active threads halves
- At the end, the reduce result will be available in the first position of the array

The number of steps of the algorithm is **logarithmic** in the size of the array (in the figure we need four steps). The kernel works with exactly one 1D block of 𝑳/𝟐 threads. Its code is shown below:
```c
__global__ void blocked_reduction(int *A, int L, int *sum)
{
	unsigned int i = 2 * threadIdx.x;
	for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		if (threadIdx.x % stride == 0) {
			A[i] += A[i + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		*sum = A[0];
	}
}
```

Each thread reads the element of A at position `i=2*threadIdx.`x and the element at position `i+stride`. At each iteration, `stride` doubles its value ($\log(L)$ **iterations**). At each iteration, only the threads with an identifier **multiple** of the current **stride** effectively contribute to the reduce. The first thread at the end writes the result in the variable `sum`.

#### Issues
There are two main issues in the previous implementation of the reduce kernel:
###### Low Warp Utilization
At each iteration, the number of used threads halves due to the conditional statement `if (threadIdx.x % stride == 0)`. However, such non-useful threads are **uniformly spread** among all warps of our kernel. We still have several warps active and resident, but with a fraction of threads doing useful work

![[Pasted image 20250531221452.png | 250]]
- Same example of the previous slide, with $L=16$ and 8 threads in our kernel
- Assume, for simplicity of the picture, 4 **threads per warp**
- Under these assumptions, our kernel is composed on one block with two warps
- Step 0, all threads work (100%)
- Step 1, four threads work only, but still spread in two warps (50%)
-  Step 2, still both warps work but with two threads only (25%)
- Step 3, one thread works in one warp (25%)
###### No Memory Coalescing
GPU hardware merges memory requests from different threads of the same warp into a **single memory transaction** to save memory bandwidth. This is possible if **consecutive threads** access **adjacent addresses** of GMEM.

![[Pasted image 20250531221724.png | 350]]

- Each thread reads two global memory locations and writes one location 
- The first read by thread `tid` is for the element `2 * tid`, the second is for the element `2 * tid + stride` where stride increases step by step (doubles)
- **Example**: thread 0 accesses element at position 0 of A, and elements at position 1, 2, 4, 8 … during the different steps

The two considered issues are the two main reasons for the non-perfect performance and GPU utilization of the kernel

### Strided Solution
To solve the two issues, we can adopt a **different assignment of threads** to the **elements of the array A**. The problem in the previous solution is that locations with partial sums become **increasingly distant** from each other. The idea is that working threads will always remain **adjacent** from each other (in terms of owned elements of A). To do that, the stride will decrease (and not increase).

![[Pasted image 20250531224800.png]]

- Same example with L = 16 and 8 threads in a single block
- Thread `tid` always reads element at position `tid` of A and the element at position `tid + stride`, where `stride` goes from 8 to 1 at each step (instead of going up as in the previous kernel version)

This new kernel (still with one **1D block only**, so working with $L \leq 2048$) is defined as follows (**it requires blocks power of two**)
```c
__global__ void strided_reduction(int *A, int L, int *sum)
{
	unsigned int i = threadIdx.x;
	for (unsigned int stride = blockDim.x; stride >=1; stride /= 2) {
		if (threadIdx.x < stride) {
			A[i] += A[i + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		*sum = A[0];
	}
}
```

The `stride` parameter is first initialized to the size of the block, then it is halved at each iteration of the for loop. As before, the first thread is in charge of writing the final result into `sum` at the end. A block-level synchronization like in the previous version is employed at each iteration of the for loop.

The previous kernel allows to improve **warp utilization** and improve **memory bandwidth** utilization. Eg, with L = 256, 128 threads (4 warps) in the first iteration work without divergence in both the kernels studied before. However, in the second iteration, 64 threads work and they are the ones in the first 2 warps in this new implementation (while they were spread in still 4 warps in the first kernel)

![[Pasted image 20250531225638.png]]

- This version improves **warp utilization** and **memory coalescing**
- Dropping threads are clustered in the smallest number of warps as possible
- Furthermore, all threads in the same warp always access elements of A at contiguous positions
- Example with L=16. At step 0, thread 0 reads `A[0]` and `A[9]`, while thread 1 reads `A[1]` and `A[10]`; at step 1, thread 0 reads `A[0]` and `A[5]` while thread 1 reads `A[1]` and `A[6]`

### SMEM Solution
We can observe that while specific data values are not reused, the same memory locations are repeatedly read and written. **Optimization**: load inputs to **[[Shared Memory (SMEM) on GPU|shared memory (SMEM)]]** first and perform the reduction tree on SMEM. Advantage is that SMEM is much faster than GMEM, therefore data reuse can be exploited to accelerate the kernel runtime.

![[Pasted image 20250531230131.png]]

- First step, each thread `tid` reads the element at position 𝒕𝒊𝒅 and the element at position `tid + stride` (with stride equal to 8) of A
- The result is written at position 𝒕𝒊𝒅 of an array in SMEM
- The remaining steps work with the array in SMEM only

The kernel using the shared memory is the following.
```c
__global__ void smem_reduction(int *A, int L, int *sum)
{
	extern __shared__ int smem_buf[];
	unsigned int i = threadIdx.x;
	smem_buf[i] = A[i] + A[i + blockDim.x];
	for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
		__syncthreads();
		if (threadIdx.x < stride) {
			smem_buf[i] += smem_buf[i + stride];
		}
	}
	if (threadIdx.x == 0) {
		*sum = smem_buf[0];
	}
}
```

Differently from the previous kernel, the input array in global memory is not modified by the kernel above. In the first step, all threads write in their position of the `smem_buf` buffer. For the remaining steps, the kernel is identical to the previous one (apart from reading/writing into SMEM instead of GMEM)

### Coarse Solution
So far, all our kernels worked with **a single 1D block**. The maximum array size is **2048 elements** and the kernels work with a block size that must be a **power of 2**. We can remove the array size constaint: each array portion (**segment**) is computed by a block independently (partial results are atomically accumulated into sum at the end).

![[Pasted image 20250531230444.png |  450]]

#### Segment Approach
Let us apply this idea on the Strided+SMEM kernel.
```c
__global__ void smem_reduction(int *A, int L, int *sum) // Strided+SMEM version
{
	extern __shared__ int smem_buf[];
	unsigned int segment = 2*blockDim.x*blockIdx.x;
	unsigned int i = segment + threadIdx.x;
	unsigned int tid = threadIdx.x;
	smem_buf[tid] = A[i] + A[i + blockDim.x];
	for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
		__syncthreads();
		if (tid < stride) {
			smem_buf[tid] += smem_buf[t + stride];
		}
	}
	if (tid == 0) {
		atomicAdd(sum, smem_buf[0]);
	}
}
```

Such a **segmented approach** can also be applied to the `blocked_reduction` and `strided_reduction` kernels to support arrays of arbitrary length.

The variable segment indicates the position in the array of the first element corresponding to the given block. Remember that the shared-memory array `smem_buf` is one per block, so `smem_buf[0]` is one different per block. The global reduce result is obtained with an `atomicAdd` performed by the first thread of each block.

In the previous kernels, each thread initially reduces two elements of the input array. Then, we start the reduction tree. With an array `A` of size `L`, we have `L/2` threads organized into `L/2B` blocks (with `B` the block size). However, a GPU has a limited number of SMs capable of running a limited number of resident blocks. The hardware **serializes** the surplus of blocks by executing a new block whenever an old one has completed.

![[Pasted image 20250531231121.png]]

If the hardware serializes additional blocks, we are better off serializing them ourselves more efficiently. 

#### Thread Coarsening
**Thread coarsening** is an optimization technique that serializes some of the work in fewer threads to reduce overheads.

![[Pasted image 20250531231253.png]]

- In the figure on the above, each segment of the array is of **32 elements**, and it is computed by **8 threads**
- Each thread performs **three steps** sequentially (without  [[Barriers]]), to aggregate four elements of the array
- Thread 0 compute `A[0] + A[8] + A[16] + A[24`] written in `smem_buf[0]`
- Then, we continue with a logarithmic number of steps as before

We observe that all threads are active in aggregating four elements by running the three steps without synchronization. The idea can be generalized by associating to each thread $2k$ elements of $A$, where $k\geq 1$Thread coarsening is enabled if $k\geq 2$

The new kernel with thread coarsening is the following:
```c
__global__ void coarse_reduction(int *A, int L, int *sum)
{
	extern __shared__ int smem_buf[];
	unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
	unsigned int i = segment + threadIdx.x;
	unsigned int tid = threadIdx.x;
	if (i < L) {
		int initial = A[i];
		for (unsigned int tile = 1; tile < COARSE_FACTOR*2; tile++) {
			initial += A[(tile * blockDim.x) + i];
		}
		smem_buf[tid] = initial;
		for (int stride = blockDim.x/2; stride >= 1; stride /= 2) {
			__syncthreads();
			if (tid < stride) {
				smem_buf[tid] += smem_buf[tid + stride];
			}
		}
		if (tid == 0) {
			atomicAdd(sum, smem_buf[0]);
		}
	}
}
```

The macro `COARSE_FACTOR` is the value $k>1$ (at least 1). The first for loop is the sequential computation of the **local reduce result** of a thread before starting the reduction tree.

Idea of the execution time breakdown
![[Pasted image 20250531232108.png | 500]]

- **left**: two blocks of **8 threads** each, computing the local reduce result on two segments of **16 elements** each of the array (total **32 elements**)
- **right**: one block of **8 threads** computing the local reduce result on a segment of **32 elements** of the array.

On the left, the execution of two blocks serialized by the hardware (**8 steps** in total). On the right, we need only **6 steps**.


### Performance Results
Tests of the different solutions of the reduce computational kernel

![[Pasted image 20250531233316.png | 400]]

Differences are remarkable with large array sizes (greater than 0.5 MiB at least). Each optimization contributes to a better performance
# Reference