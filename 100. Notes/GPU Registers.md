**Data time:** 10:44 - 31-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Execution Model]]

**Area**: [[Master's degree]]
# GPU Registers

Each Stream Multi-Processor has several thousands of **32-bit registers** that are partitioned among **resident warps**. Used for automatic scalar variables and thread coordinates. Data in the registers are **private** of the thread (scope).

**Example**. MatrixMUL with 2D grid and 2D blocks
```c
__global__ void matrix_mul(float *A, float *B, float *C, int N)
{
	int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
	if ((iy < N) && (ix < N)) {
		float val = 0;
		for (int k = 0; k<N; k++)
			val += A[(iy * N) + k] * B[(k * N) + ix];
		C[(iy * N) + ix] = val;
	}
}
```

Variables `iy, ix, val, k` are scalar and automatic, so the compiler will allocate them to registers. In the case of non-scalar automatic variables (e.g., arrays)? **Per-thread local memory** (stored in the off-chip RAM) is used.


# References