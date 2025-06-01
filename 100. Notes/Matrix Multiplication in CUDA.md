**Data time:** 12:26 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Matrix Multiplication in CUDA

Basic component of linear algebra libraries with many real-world applications (e.g., deep learning with CNN). Let $A$ be a matrix of size $N\times M$ and $B$ a matrix of size $M\times R$, the result of the matrix multiplication is a matrix $C$ of size $N\times   R$. The generic element of $C$ is equal to:
$$
C[i,j] = \sum_{k=0}^{M-1} (A[i,k] \cdot B[k,j])
$$
![[Pasted image 20250601122912.png | 350]]

- Each element of C is the result of the inner product between one row of A and one column of B
- $N \times R$ Virtual Processors each one associated with one element of C
- Each VP computes one element of C by reading one row of A and one column of B (**[[Map Parallelization|map]]**)
- Basic CUDA version with one CUDA thread per VP

### Naive Solution
Naïve kernel with $N\times R$ threads each one computing one element of $C$
```c
__global__ void naive_mm(float *A, float *B, float *C, int N, int M, int R)
{
	int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (ix < R && iy < N) {
		float val = 0;
		for (int k=0; k<M; k++) { // access row iy of A and column ix of B
			val += A[iy * M + k] * B[k * R + ix];
		}
		C[iy * R + ix] = val;
	}
}
```

This computation has an **arithmetic intensity (AI) of 0.25 operations per byte**. **Redundant accesses**: e.g., thread (0,0) and thread (1,0) accesses the same row of the matrix 𝑨 (row 0). Again, the idea is to perform **tiling**. Each tile should fit in the available shared memory of a block (a few tens of KiBs). Each tile must be computed independetly.

### Tiling Solution
###### Example
In this example, we have a **2D grid of 2D blocks**. We have 4 blocks of **2x2=4 threads** each Threads proceed with **M/TILE_DIM** iterations (4 in the figure) At each iteration, they load in SMEM **two input tiles**, one of A and one of B, and update the result of the corresponding **output tile of C**.

![[Pasted image 20250601124809.png | 600]]

![[Pasted image 20250601124845.png | 600]]

![[Pasted image 20250601124907.png | 600]]

![[Pasted image 20250601124934.png | 600]]

The kernel is shown below. We iterate among all the tiles forming a set of contiguous rows (columns) of $A(B)$ respectively.

```c
__global__ void tiled_mm(float *A, float *B, float *C, int N, int M, int R)
{
	__shared__ float A_smem[TILE_DIM][TILE_DIM];
	__shared__ float B_smem[TILE_DIM][TILE_DIM];
	int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
	float val = 0;
	// load A and B tiles
	for (int t=0; t<(M/TILE_DIM); t++) {
		A_smem[threadIdx.y][threadIdx.x] = A[(iy * M) + (t * TILE_DIM) + threadIdx.x];
		B_smem[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y)*R + ix];
		__syncthreads();
		// compute C tile
		for (int k=0; k<TILE_DIM; k++) {
			val += A_smem[threadIdx.y][k] * B_smem[k][threadIdx.x];
		}
		__syncthreads();
	}
	if (ix < R && iy < N) {
		// update C in global memory
		C[iy*R + ix] = val;
	}
}
```

Arithmetic intensity increases by a factor equal to TILE_DIM For example, with **TILE_DIM=16**, AI **becomes 4 FP/bytes** rather than 0.25.

### Performance Comparison
Experimental results with M=N=R for the sake of simplicity. The tile size (equal to the size of a CUDA block) is fixed to 32x32 (so `TILE_DIM` is 32, the maximum size, why?). Tiled version outperforms the naive one.

![[Pasted image 20250601125337.png | 400]]
### Boundary Conditions
The versions of the MM kernels were developed assuming M, N and R exact powers of two (such as the tile size). If the tile size is a power of two, but M, N and R are general values, it is possible that tiles go beyond the boundaries of the input matrices. This requires proper `if` statements in the kernel code. Try to do that as an exercise

![[Pasted image 20250601125259.png | 400]]
# References