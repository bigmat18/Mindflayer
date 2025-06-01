
**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Stencil in CUDA

Stencil-based computations often derive from numerical methods to solve partial differential equations. In computers, functions are represented with their discrete representation (e.g., a 1D function is stored in an array). Partial differential equations express the relationships between functions, variables, and their derivatives.

**Example**: 1D function $f(x)$ discretized in an array F. We would like to compute its **first-order derivative** at a point x.
$$
f'(x) = \frac{f(x+h) - f(x-h)}{2h} + O(h²) 
$$
Where the value $h$ is the spacing between neighboring points in the array. Since the grid spacing is **h**, and $x = i \cdot h$ we have:
$$
f'[i] = -\frac{1}{2h} F[i-1] + \frac{1}{2h} F[i+1]
$$
The calculation involves the current estimated function values at grid points $[i-1, i, i+1]$ with coefficients $-1/2h, 0, 1/2h$

###### Example
The previous example is a **3-point 1D stencil**. If we look for the second-order derivative, we have a **5-point 1D stencil**. Functions of two variables are discretized in both dimensions, so we have a **2D grid**. If the partial differential equation involves the first-order partial derivates by only one of the two variables, we have a **5-point 2D stencil**. If we consider functions with three variables (x, y and z), we have a **3D** representation. In many problems, such a computation is repeated **iteratively** (each iteration is called a **stencil sweep**)

![[Pasted image 20250601003819.png]]

### Stencil Swap
In stencil computations originating from **finite-difference methods**, the grid points on the boundaries will not change from input to output (in the figure they are the **white elements** on the left- and the right-hand side of the figure below)

![[Pasted image 20250601003949.png]]

Example above of a 2D grid of 16 × 16 points. Sub-matrices of 4 × 4 points are executed by a thread block. White points are not modified (**ghost cells**). We will study the case of a 3D input and output cubes
(function with three variables). This is because stencils are often applied to **high-dimensional data** (this is a main difference with other computations that might appear similar like the convolution)

![[Pasted image 20250601103859.png]]

Example above of a **3D grid** of 8 × 8 × 8 = 512 points. Each output point is computed by reading 7 input points in 3 dimensions. Again, boundaries points are not modified (**ghost cells**)

### Naive Solution
We consider the case of the **7-point** stencil in the 3D case, where each point has 6 neighbors. We focus on the code run on GPU to perform a **single stencil sweep** (more sweeps can be sequentially run on the device until convergence)

```c
__global__ void naive_stencils(float *IN, float *OUT, unsigned int N)
{
	unsigned int iz = (blockIdx.z*blockDim.z) + threadIdx.z;
	unsigned int iy = (blockIdx.y*blockDim.y) + threadIdx.y;
	unsigned int ix = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	if (iz >= 1 && iz < N-1 && iy >= 1 && iy < N-1 && ix >= 1 && ix < N-1) 
		OUT[iz*N*N + iy*N + ix] = C0 * IN[iz*N*N + iy*N + ix]
								+ C1 * IN[iz*N*N + iy*N + (ix - 1)]
								+ C2 * IN[iz*N*N + iy *N + (ix + 1)]
								+ C3 * IN[iz*N*N + (iy - 1)*N + ix]
								+ C4 * IN[iz*N*N + (iy + 1)*N + ix]
								+ C5 * IN[(iz - 1)*N*N + iy*N + ix]
								+ C6 * IN[(iz + 1)*N*N + iy*N + ix];
	}

}
```

C1-C6 are the coefficients that depend on the partial differential equation to be solved.
Each thread performs 13 FP operations and loads 7 input values of 4 bytes each. **Arithmetic intensity** is equals to **0.46**.

### Tiling Approach
The code has a low arithmitic intensity that can be improved with **tiling** (as we did for the convolution but with some differences). **Example**: 5-point stencil with 2D grid:

![[Pasted image 20250601104542.png | 550]]

In the 3D case (7-point stencil), each block has `IN_TILE_DIM³` threads of which `(ÌN_TILE_DIM - 3)³` are active in calculating the output tile elements. **Artithmetic intensity (AI)** is
$$
AI = \frac{13 \cdot (IN\_TILE\_DIM - 2)³}{4\cdot IN\_TILE\_DIM³} = 3.25 \cdot \bigg( 1 - \frac{2}{IN\_TILE\_DIM} \bigg)³ < 3.25
$$
The asymptotic evaluation shows that the benefit of using SMEM is significantly lower than that for the convolution. The tiled kernel is shown below:

```c
__global__ void tiled_stencils(float *IN, float *OUT, unsigned int N)
{
	unsigned int iz = (blockIdx.z*OUT_TILE_DIM) + threadIdx.z - 1;
	unsigned int iy = (blockIdx.y*OUT_TILE_DIM) + threadIdx.y - 1;
	unsigned int ix = (blockIdx.x*OUT_TILE_DIM) + threadIdx.x - 1;
	__shared__ float IN_smem[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
	
	if (iz >= 0 && iz < N && iy >= 0 && iy < N && ix >= 0 && ix < N) {
		IN_smem[threadIdx.z][threadIdx.y][threadIdx.x] = IN[iz*N*N+ iy*N + ix];
	}
	
	__syncthreads();
	if (iz >= 1 && iz < N-1 && iy >= 1 && iy < N-1 && ix >= 1 && ix < N-1) {
		
		if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 && threadIdx.y >=1 &&
			threadIdx.y < IN_TILE_DIM-1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM-1) {
			
			OUT[iz*N*N + iy*N + ix] = C0 * IN_smem[threadIdx.z][threadIdx.y][threadIdx.x]
									+ C1 * IN_smem[threadIdx.z][threadIdx.y][threadIdx.x-1]
									+ C2 * IN_smem[threadIdx.z][threadIdx.y][threadIdx.x+1]
									+ C3 * IN_smem[threadIdx.z][threadIdx.y-1][threadIdx.x]
									+ C4 * IN_smem[threadIdx.z][threadIdx.y+1][threadIdx.x]
									+ C5 * IN_smem[threadIdx.z-1][threadIdx.y][threadIdx.x]
									+ C6 * IN_smem[threadIdx.z+1][threadIdx.y][threadIdx.x];
			
			}
		}
}
```

#### Issues
Asymptotically, we can achieve an arithmetic intensity of **3.25 FP operations per byte** read from GMEM. Unfortunately, the hard limit to the block size makes things more complicated. A block of $8\times 8 \times 8 = 512$ a practical limit. Therefore, AI is practically much lower (**1.37**).

For the [[Convolution in CUDA]] problem, tiling is more effective in increasing AI. Why? Observe that in general **halo elements** are less used than **non-halo elements** in each input tile.

**Example**
![[Pasted image 20250601110208.png | 550]]

### Thread Coarsening
##### Challenge 1: Block size limit
the input tile size in the current implementation is the same as the block size, which is limited by the hardware (1024 threads at most).

The **solution** is: each output tile is going to be computed by a block having a number of threads equal to a **x-y plane** of the input tile only. Therefore, each thread is now responsible to compute more output elements of the output tile.

![[Pasted image 20250601111739.png | 550]]

Each **x-y plane** of the input tile consists of $6² = 36$ **points**, while each x-y plane of the **output tile** is of $4^2=16$ **points**. Related to the picture, the kernel is a 3D grid of size (2,2,2) for a total of 8 blocks. We have **2D blocks of (6,6) threads** each (36). Totally, we have **8 blocks of 36 threads** (288 threads totally and not 1728 as before)

##### Challenge 2: SMEM capacity limit
the input tile size in the current implementation determines the SMEM usage per block

The **solution** is each block computes one x-y plane of the output tile at a time, more iterations are needed to compute a full output tile. Therefore, each block needs only **three x-y planes** of the input tile to be stored in SMEM at a time. This technique is called **slicing**

![[Pasted image 20250601111959.png | 550]]

In the **non-tiled implementation**, each block needed a SMEM buffer of **216 floats (864 bytes)**. With the new idea, each block stores in SMEM **three x-planes** of 36 floats each (totally we need **432 bytes** per block)

##### Slicing
Each block has 36 threads. It executes `OUT_TILE_SIZE` iterations to compute, at each iteration, one x-y plane of the output tile (each output plane is of 16 elements). Each block brings in SMEM three **x-y planes** of the input tile at a time (each input plane is of **36 elements**)

![[Pasted image 20250601112351.png]]

**Iteration 1** is not shown for graphical reasons with the pictures (you can imaging it). Elements at position (\*,\*,z) with z fixed are calculated by the **same thread**.

For example, we use a block of 32x32=1024 threads. At each iteration, 1024 threads write an output plane of 900 elements and this proceeds for 30 iterations (27000 elements totally)

```c++
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM-2)
__global__ void coarsed_stencils(float *IN, float *OUT, unsigned int N)
{
	int zStart = blockIdx.z*OUT_TILE_DIM;
	int iy = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
	int ix = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
	__shared__ float prev_smem[IN_TILE_DIM][IN_TILE_DIM];
	__shared__ float curr_smem[IN_TILE_DIM][IN_TILE_DIM];
	__shared__ float next_smem[IN_TILE_DIM][IN_TILE_DIM];
	…
```

The coordinate $(iy, ix)$ identifies the point to be calculated in the current output tile plane by the thread, while `zStart` is the current coordinate in the z axis. All points calculated by the threads in the kernel at the same iteration will have the same coordinate `zStart`. We allocate space for three neighbor input tile planes in SMEM.

Each block loads into the SMEM the three **input tile planes** that contain all the points that are needed to calculate the values of the **current output tile plane**.

**Example** with `IN_TILE_DIM` of 6 and `OUT_TILE_DIM` of 4. 8 blocks of 36 threads each. Threads in a block compute all the 64 elements of the output tile by writing 16 elements per x-y plane at each iteration (four iterations totally).

![[Pasted image 20250601121547.png | 380]]

![[Pasted image 20250601121619.png | 380]]

![[Pasted image 20250601121639.png | 380]]

![[Pasted image 20250601121703.png | 380]]

![[Pasted image 20250601121724.png | 380]]


Each block loads into the SMEM the three input tile planes that contain all the points that are needed to calculate the values of the current output tile plane.

```c
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM-2)
__global__ void coarsed_stencils(float *IN, float *OUT, unsigned int N)
{
	…
	if (zStart-1 >=0 && zStart-1 < N && iy >= 0 && iy < N && ix >= 0 && ix < N) {
		prev_smem[threadIdx.y][threadIdx.x] = IN[(zStart - 1)*N*N + iy*N + ix];
	}
	if (zStart >=0 && zStart < N && iy >= 0 && iy < N && ix >= 0 && ix < N) {
		curr_smem[threadIdx.y][threadIdx.x] = IN[zStart*N*N + iy*N + ix];
	}
	for (int z = zStart; z < zStart + OUT_TILE_DIM; z++) {
		if (z+1 >=0 && z+1 < N && iy >= 0 && iy < N && ix >= 0 && ix < N) {
			next_smem[threadIdx.y][threadIdx.x] = IN[(z+1)*N*N + iy*N + ix];
		}
		__syncthreads();
	…
```

During the first iteration, all threads in a block collaborate to load the third layer needed for the **current output tile plane** into the SMEM array `next_smem`. 

Each thread calculates its output point in the current output tile plane using the four x-y neighbors stored in `curr_smem`, the z neighbor in `prev_smem`, and the z neighbor in `next_smem`.

```c
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM-2)
__global__ void coarsed_stencils(float *IN, float *OUT, unsigned int N)
{
	…
	for (int z = zStart; z < zStart + OUT_TILE_DIM; z++) {
		…
		if (z >= 1 && z < N-1 && iy >= 1 && iy < N-1 && ix >= 1 && ix < N-1) {
			if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1
			&& threadIdx.x < IN_TILE_DIM - 1) {
				OUT[z*N*N + iy*N + ix] = C0*curr_smem[threadIdx.y][threadIdx.x] +
				C1*curr_smem[threadIdx.y][threadIdx.x-1] +
				C2*curr_smem[threadIdx.y][threadIdx.x+1] +
				C3*curr_smem[threadIdx.y+1][threadIdx.x] +
				C4*curr_smem[threadIdx.y-1][threadIdx.x] +
				C5*prev_smem[threadIdx.y][threadIdx.x] +
				C6*next_smem[threadIdx.y][threadIdx.x];
			}
		}
		__syncthreads();
		prev_smem[threadIdx.y][threadIdx.x] = curr_smem[threadIdx.y][threadIdx.x];
		curr_smem[threadIdx.y][threadIdx.x] = next_smem[threadIdx.y][threadIdx.x];
	}
}
```

### Register Tilining
**Observation**: in the calculation of the current output plane, each element of prev_smem and next_smem is accessed by one thread only

![[Pasted image 20250601122116.png]]

**Consequence**: the z neighbors in `prev_smem` and `next_smem` can instead stay in the registers of that thread. New kernel design:
- The initial loading of the **previous** and **current input tile planes** and the loading of the **next plane** of the input tile before each new iteration are all performed with register variables as destination.
- In addition, the kernel always maintains a copy of the current plane of the input tile in the shared memory, That is, the x-y neighbors of the active **input tile** plane are always available to all threads that need to access these neighbors

```c
__global__ void coarsed_tiled_stenciles(float *IN, float *OUT, unsigned int N)
{

	int zStart = blockIdx.z * OUT_TILE_DIM;
	int iy = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
	int ix = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;
	float prev;
	__shared__ float curr_smem[IN_TILE_DIM][IN_TILE_DIM];
	float curr;
	float next;
	
	if (zStart-1 >=0 && zStart-1 < N && iy >= 0 && iy < N && ix >= 0 && ix < N) {
		prev = IN[(zStart - 1)*N*N + iy*N + ix];
	}
	
	if (zStart >=0 && zStart < N && iy >= 0 && iy < N && ix >= 0 && ix < N) {
		curr = IN[zStart*N*N + iy*N + ix];
		curr_smem[threadIdx.y][threadIdx.x] = curr;
	}
	
	for (int z = zStart; z < zStart + OUT_TILE_DIM; z++) {
		if (z+1 >=0 && z+1 < N && iy >= 0 && iy < N && ix >= 0 && ix < N) {
			next = IN[(z+1)*N*N + iy*N + ix];
		}
		__syncthreads();
		
		if (z >= 1 && z < N-1 && iy >= 1 && iy < N-1 && ix >= 1 && ix < N-1) {
			if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
				threadIdx.x < IN_TILE_DIM - 1) {
				OUT[z*N*N + iy*N + ix] = C0*curr
										+ C1*curr_smem[threadIdx.y][threadIdx.x-1]
										+ C2*curr_smem[threadIdx.y][threadIdx.x+1]
										+ C3*curr_smem[threadIdx.y+1][threadIdx.x]
										+ C4*curr_smem[threadIdx.y-1][threadIdx.x]
										+ C5*prev
										+ C6*next;
			}
		}
		__syncthreads();
		prev = curr;
		curr = next;
		curr_smem[threadIdx.y][threadIdx.x] = next;
	}
}
```
# References