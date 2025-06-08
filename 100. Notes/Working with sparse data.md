**Data time:** 21:43 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Working with sparse data

In a sparse matrix, the majority of elements are **zeros**. Storing and computing them is a waste of **memory occupation**, **bandwidth**, and **computation time**. 

Proper data formats (applying a sort of compression) are needed. However, this often generates **irregularities** that might lead to a **sub-utilization** of the **GMEM bandwidth**, more **warp divergence**, and **load imbalance** in general.

![[Pasted image 20250601214532.png | 200]]

Many **data formats** for sparse matrices
- Coordinate Format (COO)
- Compressed Sparse Row (CSR)
- ELLPACK Format (ELL)
- Jagged Diagonal Storage (JDS)

Format design considerations
- **Space efficiency** (memory consumed)
- **Flexibility** (ease of adding/reordering elements)
- **Accessibility** (ease of finding desired data)
- **Memory access pattern** (coalescing?)
- **Load balance** (control divergence?)

We study the **sparse matrix-vector product problem (SpMV)**. We suppose the **matrix sparse**, while the vector is **dense**

![[Pasted image 20250601215052.png |400]]
### [[Coordinate Format (COO)]]

### [[Compressed Sparsed Row (CSR)]]

### [[ELL Data Layout]]

### [[JDS Format ]]


# References