**Data time:** 23:36 - 31-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Convolution in CUDA

The **definition** of the problem is an operator where each output element is the weighted sum of the corresponding input element and a collection of input elements centered on it. The array of weights is called **convolution filter** and its size can change depending on the problem. **Different dimensionalities**: 1D (audio), 2D (image), 3D (video).

**Example (1D):** x is the input array (signal), y the output array (signal), while f is the convolutional filter. Below an example of calculation of the element at **position 2**:

![[Pasted image 20250531233851.png | 550]]

Calculation of the element at **position 3**:
![[Pasted image 20250531234049.png |550]]

Calculation of the element at **position 1**. The filter has **radius 2**, so we need to access element at **position -1** that does not exist.

![[Pasted image 20250531234157.png]]

Typical approach is to assign a default value (e.g., 0) to the missing elements (**ghost cells**). The same idea can be applied to a 2D problem (image).

![[Pasted image 20250531234245.png | 450]]

Again, if we access positions that do not exist, we consider the presence of fictitious cells (ghosts) with default values (e.g., 0).

![[Pasted image 20250531234320.png | 450]]

### Naive Solution
Assign one thread to compute each output element by looping over the input elements and filter weights. **Example 2D** (derive 1D and 3D cases as an exercise). It is convenient to organize the kernel as a **2D grid** of **2D blocks**.

![[Pasted image 20250531235204.png | 450]]

Each CUDA thread is a **VP** assigned to more elements of A (in the figure 25 inputs) and one element of B (output). The naïve kernel computes each element of the output matrix B by reading the corresponding element of the input matrix A and the neighbors depending on the radius r. The kernel is shown below:

```c
__global__ void naive_conv(float *A, float *B, float *F, int w, int h)
{
	int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (ix >=0 && ix < w && iy >=0 && iy < h) {
		for (int fy=0; fy < 2*RADIUS+1; fy++) {
			for(int fx=0; fx < 2*RADIUS+1; fx++) {
				int iyy = iy - RADIUS + fy;
				int ixx = ix - RADIUS + fx;
				if (iyy >= 0 && iyy < h && ixx >= 0 && ixx < w) {
					B[iy * w + ix] += F[fy * (2*RADIUS + 1) + fx] * A[iyy * w + ixx];
				}
			}
		}
	}
}
```

Boundary conditions implemented by `if` statements before the two for loops (to check the boundaries of the input A) and within them (for the neighboring cells related to point $ix, iy$).
#### Issues
The previous kernel has two main issues:
- **Control flow divergence**: threads that access ghost elements take different decision in the if stamements, and this results in thread divergence
- **Memory bandwidth**: the problem is memory intensive since the ratio between floating-point operations and global memory accesses (**arithmetic intensity**, shortly **AI**) il quite low

The **first issue** is not a major shortcoming, since only boundary elements need ghost cells. So, for large inputs, the impact of control flow divergence should be relatively small. The **second issue** is more critical: only 2 FP operations for every 8 bytes read from the memory (ratio is 0.25).

So, this second is the most critical issue. **We need to reduce the impact of memory accesses**. We can follow two strategies (applied simultaneously in the same code):
- Using the **[[Shared Memory (SMEM) on GPU|shared memory]]** whenever it is possible to avoid reading from GMEM multiple time the same elements of A
- Using the **constant memory** to mitigate the accesses to F.

### Tiled Convolution
We exploit a tiled approach to leverage shared memory in the kernel processing and, so, to obtain better performance. **Definition**: an **input tile** is a collection of input elements that are needed to compute a collection of output elements called **output tile**.

![[Pasted image 20250531235938.png | 550]]

The input tile has size `IN_TILE_DIM²`, while the output tile is `OUT_TILE_DIM²`, with `IN_TILE_DIM = OUT_TILE_DIM + 2r`.

- The idea is that threads collaborate to first load in SMEM all the elements of the input tile (**phase 1**)
- Then, they start computing the elements of the corresponding output tile (**phase 2**)

How to deal with the asymmetry between the input and the output tile sizes? Two ways:
- **Block size maching the input tile size:** each block matches the size of the input tile. All threads load 1 element of the input tile (**phase 1**) to copy it in SMEM. Some threads will be disabled during the computation of the elements in the output tile (**phase 2**)

- **Block size matching the output tile size**: each block matches the size of the output tile. Some threads will be responsible for copying more elements of the input tile into SMEM (**phase 1**) while all of them compute one element of the output tile (**phase 2**)

Below we will follow the first strategy to design our kernels since this is easier to apply.

The block size is equal to the input tile size (64 threads in the figure below). Each thread is assigned to one element of the input tile (**light-blue elements** are halo elements, **dark-blue ones** are non-halo elements that directly correspond to one element of the output tile each).

![[Pasted image 20250601001247.png]]

1. All the 64 threads of the block read their elements of the input tile into an SMEM buffer (of the same size of the input tile)
2. 16 threads of the block (the internal ones) compute the convolution by reading the input tile in SMEM with halo cells, the filter, and write results in the output tile

Let us exemplify the idea with a smaller example with $L=4$ so with input and output matrices A and B of size $L\times L = 16$. Let us use an output tile size of 4 (and a filter **radius** of 1). Therefore, the input tile size is of 16 elements. We launch a CUDA kernel that consists of a **2D grid** of **2D blocks**. We have **4 blocks of 16 threads each (4x4)**.

![[Pasted image 20250601002058.png]]

The kernel of the tiled 2D convolution is shown below. We also accumulate the result on a local variable `pvalue` allocated in a register (to further reduce accesses to GMEM).

```c
__global__ void tiled_conv(float *A, float *B, float *F, int w, int h)
{
	// loading the input tile
	__shared__ float A_smem[IN_TILE_DIM][IN_TILE_DIM];
	int ix = (blockIdx.x * OUT_TILE_DIM) + threadIdx.x - RADIUS;
	int iy = (blockIdx.y * OUT_TILE_DIM) + threadIdx.y - RADIUS;
	if (ix >= 0 && ix < w && iy >=0 && iy < h) {
		A_smem[threadIdx.y][threadIdx.x] = A[iy * w + ix];
	}
	else {
		A_smem[threadIdx.y][threadIdx.x] = 0.0;
	}
	__syncthreads();
	
	int tileX = threadIdx.x - RADIUS;
	int tileY = threadIdx.y - RADIUS;
	// writing the output tile
	if (ix >= 0 && ix < w && iy >=0 && iy < h) { // active thread
		if (tileX >= 0 && tileX < OUT_TILE_DIM && tileY >=0 && tileY < OUT_TILE_DIM) {
			float pvalue = 0;
			for (int fy=0; fy<2*RADIUS+1; fy++) {
				for(int fx=0; fx<2*RADIUS+1; fx++) {
					pvalue += F[fy * (2*RADIUS + 1) + fx] 
							* A_smem[(tileY + fy)][(tileX + fx)];
				}
			}
			B[iy * w + ix] = pvalue;
		}
	}
}
```

### Constant Tiled
The filter array has interesting properties:
- it is small compared with the input data size
- it is used in read-only mode
- all threads access it in the same order

So, the filter is a good candidate to be located in constant memory. The potential advantage is to exploit the optimized caching performed by GPU on constant memory. To declare and initialize constant memory for our convolution kernel we have to:

```c
// declaration of the filter in constant memory (global variable)
__constant__ float dev_F[2*RADIUS+1][2*RADIUS+1];

cudaMemcpyToSymbol(dev_F, host_F, (2*RADIUS+1)*(2*RADIUS+1)*sizeof(float));
```

We have further improved the **arithmetic intensity (AI)**, since we avoid reading GMEM for the filter values.

### Arithmetic Intensity
Let `IN_TILE_DIM` and `OUT_TILE_DIM` defined as before, and `RADIUS` the radius of the constant filter matrix. The **arithmetic intensity** (ratio between floating-point operations and read bytes) is measured as:

![[Pasted image 20250601002713.png]]

As an asymptotic analysis, we can assume that `OUT_TILE_DIM>> RADIUS`. Therefore the ratio becomes $(2\cdot RADIUS + 1)² / 2$. Of course, we are limited by the number of threads per block (1024). So, $IN\_TILE\_DIM \leq 32$ and $OUT\_TILE\_DIM \leq 32 - 2r$.


# References