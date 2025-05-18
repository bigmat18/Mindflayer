**Data time:** 22:25 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: 
# K-ary N-mesh Networks (Meshes)

This type of network is a **Direct Network** also called **Meshes**. They are n-dimensional grids with k nodes in each dimension. They interconnect $N = k^n$ nodes with a **node degree** of $2n$. With $n > 1$ it is possibile to have a distinct **radix** (ie number of switches/nodes) for each dimension. For example (y,x)-ary, 2-mesh.

- **Maximum Bandwidth** is O(N)
- **Latency** $O(k\cdot n)$: $O(\sqrt[n]{N})$ for arrays, grids and cubes, $O(\log_k N)$ for hypercubes. Networks with **path diversity** and **blocking**
- **Diameter** is $n(k-1) + 1$
- **Bisection width** $k^{n-1}$

![[Pasted image 20250518223109.png | 500]]
# References