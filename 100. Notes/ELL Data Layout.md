**Data time:** 22:06 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# ELL Data Layout

New data format called **ELL** (**Elias-Lemke-Lewis**, from the ELLPACK library where it was introduced)

![[Pasted image 20250601220716.png | 550]]

Each row of the `columnIdx` matrix contains the indexes of the columns in that row that contain non-zero elements.

The matrices `columnIdx` and value are padded to the size of the row with the maximum number of non-zero elements. The matrices are rectangular (proper padding is added). We can now lay the padded matrix out in **column-major order** including padding elements.

![[Pasted image 20250601221025.png | 500]]

- First element of row 𝒓 is found at `value[r]`
- Moving from one element of row r to the next one in the same row is done by adding `nRows` (number of rows of the matrix)
- Element `value[i]` is the nonzero of column `columnIdx[i]` and row `i%nRows`

Like with [[Compressed Sparsed Row (CSR)|CSR]], each CUDA thread is assigned to a different row of the matrix (**control divergence** likely happens). However, adjacent CUDA threads working on different consecutive rows access consecutive elements of `columnIdx` and value arrays (**memory coalescing**). Again, each thread computes the dot product between its row and the array b by computing one element of the array c.

![[Pasted image 20250601221244.png | 500]]

The new data format (ELL) can be implemented by a struct as follows

```c
struct ell_matrix
{
	int numRows;// number of rows of the matrix
	int max_nz_row;// max no. of nonzeros in a row
	int *non_zeros;// number of non-zero elements per row
	int *columnIdx;// device array of column indexes
	float *value;// device array of FP values
};
```

The ELL kernel is shown in the code below
```c
__global__ void ell_spmv(ell_matrix *A, float *b, float *c)
{
	unsigned int row = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (row < A->numRows) {
		float sum = 0.0f;
		for (unsigned int i = 0; i < A->non_zeros[row]; i++) {
			unsigned int pos = i*A->numRows + row;
			unsigned int col = A->columnIdx[pos];
			float value = A->value[pos];
			sum += value * b[col];
		}
		c[row] = sum;
	}
}
```

We compute `pos` as the index of `value` where to find the i-th non-zero element of row `row`.

### COO + ELL Format
ELL has a **low space efficiency** and **control divergence** that impairs performance. An alternative approach is based on a combination of **[[Coordinate Format (COO)|COO]]** and ELL, to reduce **padding elements** in ELL and **control divergence**

![[Pasted image 20250601221620.png | 600]]

For this reason, we can remove the last three nonzeros of row 1 and the last two nonzeros of row 6 from the ELL representation. Such elements are represented by an **additional COO representation**.

The kernel operates in two phases (ELL+COO). Both are **memory coalesced** and with good **control divergence** now. **Space efficiency** has been mitigated. However, we still use **atomic instructions** for the COO part.
# References