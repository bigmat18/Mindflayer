**Data time:** 17:06 - 13-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]] [[Parametrization Techniques]]

**Area**: [[Master's degree]]
# Harmonic Parametrization

With this parametrization we have a Linear System. A sparse Matrix (2n x 2n) where n is number of vertices of the mesh. Express each point as weighted sum of its neighbors. Find x such that $Ax = 0$. At the end $x$ are the final UV coordinates.

In other words we want found a solution su Laplace Equation:
$$
\Delta f = 0
$$
where $\Delta$ is the [[Discrete Differential Operators|Laplace operator]] on the mesh.

 - To apply this technique we need to fix the boundary of the mesh to UV (we map boundary on unique circle)
 - Express each UV position as linear combination of neighbors
 - Use [[Mass-Spring Parametrization|cotangent weights]] (laplacian in a discrete domain)

![[Pasted image 20250413172427.png | 500]]

This approch (called **harmonic mapping**) is used to smoothly interpolate scalar values over a mesh given some sparse constraint. This because we have a system, each **vertex is a linear combination of neighbors**, for some vertex we know the value, we resolve the system  and we found the interpolations.

![[Pasted image 20250413172842.png]]

Its also useful to interpolate deformations. 

![[Pasted image 20250413173019.png | 300]]


# References