**Data time:** 20:44 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Matrix Transponse in CUDA

**Problem**: given a matrix $A \in \mathbb{R}^{N\times M}$ we want to compute its transpose $A^T \in \mathbb{R}^{M\times N}$. Suppose below N=3 and M=4.

![[Pasted image 20250601204716.png | 550]]

### Naive Solution
A naïve kernel assigns each CUDA thread to an element of 𝑨, which is reponsible to copy it in one element of $A^T$. What about the **alignment** and the **coalescing** of accesses to GMEM by warps with this approach? Threads read the matrix A in an aligned and coalesced manner. However, the writings on $A^T$ are **not** coalesced.

The **dual approach** assigns each CUDA thread to one element of $A^T$ (writes to $A^T$ are now coalesced but readings not). **Naïve CUDA kernel** to compute the transpose given the input matrix (each thread is assigned to an element):

```c
__global__ void naive_transpose(float *A, float *T, int N, int M)
{
	int ix = (blockDim.x * blockIdx.x) + threadIdx.x;
	int iy = (blockDim.y * blockIdx.y) + threadIdx.y;
	if (ix < M && iy < N) {
		// original A: N rows, M columns
		T[(ix * N) + iy] = A[(iy * M) + ix];
	}
}
```

Kernel layout is a 2D grid of 2D blocks of BxB threads each. As usual, we extract the indexes of the column (`ix`) and of the row (`iy`). We consider **boundary conditions** (the kernel above runs correctly with any value of N and M). In the kernel, each CUDA thread reads one element of A (in a **coalesced manner**) and writes one element of the transpose (**not coalesced**).

### Corner Turning Solution
It is an optimization where we use the SMEM to avoid uncoalesced accesses to the GMEM.

![[Pasted image 20250601205304.png | 550]]

Threads of a block read a block of A. Threads in the same warp access elements of A in the same row within the block. Such threads copy their elements of 𝑨 into their SMEM block. Threads of a block read the elements from the SMEM buffer and copy them into the output block of $A^T$. Threads in the same warp write elements of the same **row** of the output block by reading a **column** of their SMEM block.

Version exploiting shared memory and the corner turning optimization. We use **square blocks** of `BLOCK_DIM²` threads.

```c
__global__ void smem_transpose(float *A, float *T, int N, int M)
{
	__shared__ float smem[BLOCK_DIM][BLOCK_DIM];
	int ix = (blockDim.x * blockIdx.x) + threadIdx.x;
	int iy = (blockDim.y * blockIdx.y) + threadIdx.y;
	if (ix < M && iy < N) {
		smem[threadIdx.y][threadIdx.x] = A[(iy * M) + ix];
	}
	__syncthreads();
	
	int sy = (blockIdx.x * blockDim.x) + threadIdx.y;
	int sx = (blockIdx.y * blockDim.y) + threadIdx.x;
	if (sy < M && sx < N) {
		T[(sy*N) + sx] = smem[threadIdx.x][threadIdx.y];
	}
}
```

In the first phase, threads read the elements of A (**coalesced**) and write their block in SMEM. In the second phase, threads read the columns of their block in SMEM and write them in the final traspose (**coalesced**).

### Performance Results
Although we perform additional copies (GMEM -> SMEM; SMEM -> GMEM) we achieve better bandwidth (i.e., lower kernel execution time) for large enough matrices. Results (considering 2D blocks of 16 × 16 = 256 threads)

| N, M<br>   | Naïve<br>kernel time(usec) | SMEM<br>kernel time<br>(usec) |
| ---------- | -------------------------- | ----------------------------- |
| 1024, 1024 | 199.473                    | 166.592                       |
| 8192, 8192 | 1835.9                     | 1387.33                       |
Significant improvement. The kernel execution time reduced by **16%** and **24%** compared with the naive version. The example highlights why coalescing accesses to GMEM is pivotal. Although GMEM is a **HBM** (with hundreds of GiB/s or even TiB/s), the huge parallelism of the GPU requires a careful utilization of GMEM transactions.


# References