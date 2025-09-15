**Data time:** 15:04 - 12-08-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Remeshing. Mesh Simplification and Approximation]]

**Area**: [[Master's degree]]
# Simplification Introduction

You starting discretizzazione is too dense, drop less useful information. Reduce the amount of polygons composing a mesh with minimal effect on the geometry.

![[Pasted image 20250403171818.png | 400]]

Erase redundant information with minimal effect on the geometry (in case of iso-surface).

![[Pasted image 20250403172025.png | 400]]

This is useful to reduce complexity for rendering use case. For example in case of **multi-resolution hierarchies** for efficient geometry processing, or level-of detail (LOD) rendering.

Complexity and accuracy is non a linear relation. If a rappresentation is very complicated we can discard many element and maintain a right accuracy, after a point the simplification reduce a lot the quality of the mesh:

![[Pasted image 20250403172612.png | 400]]

### Problem Statement
In this problem we have a mesh M=(V,F) and we must find a new mesh M'=(V', F') such that:
- |V'| = n < |V| and ||M - M'|| is minimal, or
- ||M - M'|| <  $\epsilon$ and |V'| is minimal

Reduce the number of vertices, minimizing the **[[Hausdorff Distance| approximating error]]** or keep below the **error** a threshold and minimize the number of vertices.

![[Pasted image 20250403174131.png | 500]]

### [[Vertex Clustering]]

### [[Incremental methods]]

### [[Variational Shape Approximation]]

### [[Greedy Shape Approximation]]


# References