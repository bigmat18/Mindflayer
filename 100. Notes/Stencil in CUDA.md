**Data time:** 00:31 - 01-06-2025

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

Example above of a 2D grid of 16 × 16 points. Sub-matrices of 4 × 4 points are executed by a thread block. White points are not modified (**ghost cells**).
# References