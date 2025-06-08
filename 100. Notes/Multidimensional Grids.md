**Data time:** 20:20 - 28-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to CUDA]]

**Area**: [[Master's degree]]
# Multidimensional Grids

The dimensionality of the grid and the blocks usually depends on the problem data characteristics. So, the reason for choosing 1D, 2D, and 3D grids and blocks is **convenience** mostly. Suppose to work with a matrix (so a 2D data structure). The matrix is always flattened in memory. For each element, we can identify:
- A pair of **logical coordinates** $(ix, iy)$ of the data element
- a linear offset $idx$ of the element $(ix, iy)$ in memory

![[Pasted image 20250528202839.png]]

Suppose the problem working with a matrix is executed on GPU with a kernel. A natural idea is to organize a **2D grid** of **2D blocks**. Each thread is assigned to one element of the matrix (i.e., it is a **map**) However, each thread must compute the linear offset of the element to find it in memory.

![[Pasted image 20250528203404.png | 450]]

For example, if each block is **16x16**, and we are looking for the element identified by the red dot located at position **(15, 2)** within the block **(1,1)**, this corresponds to the coordinates **(31, 18)** in the whole grid. The linear offset of the element, supposing the matrix represented per row in global memory (default), is:

```c
idx = iy * (gridDim.x * blockDim.x) + ix
```

### [[MatrixSUM]]

### [[ImageFLIP]]

### [[ImageBLUR]]


# References