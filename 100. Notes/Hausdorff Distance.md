**Data time:** 14:00 - 12-08-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Remeshing. Mesh Simplification and Approximation]]

**Area**: [[Master's degree]]
# Hausdorff Distance
Quantifies the notion of "similarity" is not a easy task. We can have two kinds of similarity:
- Geometric similarity (surface deviation).
- Appearance similarity (material, normal).
##### Appearance similarity
Difference between two images (trivial way):
$$D(I_1, I_2) = \frac{1}{n^2} \sum_x\sum_y d(I_1(x,y), I_2(x,y))$$
Difference between two objects: integrate the above over all possible views. This is a tecquique that have problems because there are a lot of factor, like illuminations, that can create a not valid value.

![[Pasted image 20250403180040.png | 500]]
##### Geometry similarity
To calculate this similarity we nee two main components:
- Distance function
- Function norm:
	- $L_2$: average deviation
	- $L_{\inf}$: maximum deviation - **Hausdorff distance**. That, for its definition is not a distance because is not symmetry (that is one of the main property for a distance)

###### Hausdorff distance
It's defined by the following formula:
$$D_H(S_1, S_2) = \max_{x \in S_1}(\max_{x \in S_2} D(x,y))$$
![[Pasted image 20250403180856.png | 500]]
We take each points in $S_1$ and for each points we search the closest point in $S_2$. The **Symmetric version** is the following:
$$D(S_1, S_2) =\max\{D_H(S_1,S_2),D_H(S_2,S_1)\}$$
With this version leads to a different results. This means that the two distance **are not symmetric**. Approximate as:
1. Sample one surface (uniformly distributed)
2. For each point compute $\max_{y\in S_2}D(x,y)$

This problem is **NP-hard**. It is NP-hard to decide if a given surface of n vertexes can be $\epsilon$-approximated with a surface composed by k vertices.

But even the 2D version of the problem is NP-Hard: Simplifying a polyline to k vertexes so that it $\epsilon$-approximate a optimal simplification using the undirected Hausdorff distance is NP-hard. The same holds when using the directed Hausdorff distance from the input to the output polyline, whereas the reverse can be computed in polynomial time.
# References