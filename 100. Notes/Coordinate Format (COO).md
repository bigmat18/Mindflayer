**Data time:** 21:49 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Coordinate Format (COO)

We start with a data format called **Coordinate Format (COO)**. Non-zero elements are stored in one-dimensional arrays with their **column** and **row** indexes (**structure of arrays**).

![[Pasted image 20250601215216.png]]

We can assign a **CUDA thread** to each non-zero input element of the matrix. Each CUDA thread reads one **non-zero element** `i,j` of A, the element of b at position k, and accumulates the result in `c[i]`

![[Pasted image 20250601215407.png | 500]]

Threads assigned to the elements of the same row of the matrix will update (**atomically**) the same element of the output array.

The sparse matrix is represented by the `coo_matrix` struct
```c
struct coo_matrix
{
	int non_zeros;  // number of non-zero elements
	int *rowIdx;    // device array of row indexes
	int *columnIdx; // device array of column indexes
	float *value;   // device array of FP values
};
```

The kernel is described below
```c
__global__ void coo_spmv(coo_matrix *A, float *b, float *c)
{
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < A->non_zeros) {
		unsigned int row = A->rowIdx[tid];
		unsigned int col = A->columnIdx[tid];
		float value = A->value[tid];
		atomicAdd(&c[row], value*b[col]);
	}
}
```
The atomic operation `atomicAdd` is used for the accumulation, because multiple CUDA threads update the same element of the output array (i.e., VPs do not respect the **OCR**)

###### Pros
- we can process elements in any order (no **control divergence**)
- accesses to arrays `rowIdx`, `columnIdx` and `value` are **coalesced**
###### Cons
- We need atomic instructions to accumulate results in the output array, since each thread is assigned to a pair of elements of A and b only
# References