**Data time:** 21:59 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Compressed Sparsed Row (CSR)

Grouping non-zeros elements in the same row can be done with an alternative data format for the sparse matrix called **Compressed Sparsed Row (CSR)**.

![[Pasted image 20250601220006.png | 500]]

Example above with a row without non-zero elements. Consecutive entries of `rowPtr` contain the same offset.

To perform SpMV in parallel with the CSR format, we can assign each **CUDA thread to a row of the sparse matrix**. Each CUDA thread loops for all the non-zero elements of its assigned row to perform the dot product, and accumulate the result in the same position of the output array. The accumulation **does not require atomic instructions**

![[Pasted image 20250601220313.png | 600]]

With this data format, CUDA threads correspond to our VPs (first part of the course). The definition respects the **owner compute rule (OCR)**. This avoids atomic instructions.

The new data format (**CSR**) can be implemented by a struct as follows.
```c
struct csr_matrix
{
	int numRows; // number of rows of the matrix
	int non_zeros; // number of non-zero elements
	int *rowPtr; // device array of row offsets
	int *columnIdx; // device array of column indexes
	float *value; // device array of FP values
};
```

The CSR kernel is shown in the code below
```c
__global__ void csr_spmv(csr_matrix *A, float *b, float *c)
{
	unsigned int row = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (row < A->numRows) {
		float sum = 0.0f;
		for (unsigned int i = A->rowPtr[row]; i < A->rowPtr[row+1]; i++) {
			unsigned int col = A->columnIdx[i];
			float value = A->value[i];
			sum += value * b[col];
		}
		c[row] = sum;
	}
}
```

CSR is more **space efficient** than [[Coordinate Format (COO)|COO]] although less flexible (it is difficult to add a further non-zero element later). Accesses to non-zero elements of the same row are easy with CSR. 

However, accesses are **not coalesced** (threads with contiguous identifiers access elements of value far away). Furthermore, **control divergence** likely happens (each thread iterates for a different number of non-zero elements in its row).

# References