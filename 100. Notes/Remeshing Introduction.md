**Data time:** 21:39 - 13-11-2024

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]]

**Area**: [[Master's degree]]
# Remeshing Introduction

Any discretization is an approximation of an ideal shape. For the same abstract shape we can have many different discretizations. No absolute ideal discretization exist. **Remeshing** is concerned with obtaining many different discretization with different property. For example:
- **Closeness/Distance**: how far is my discretization from the intended shapes.
- **Conciseness**: Number of primitive really needed.
- **Shape/Robusteness**: Not all triangles are equals.

## Refinement / Subdivision
We pass from a coarse object to an object more refined. Subdivision defines a smooth curve or surface as the limit of a sequence of successive refinements.

![[Pasted image 20250328141946.png | 350]]

This concept is born from the curve concepts. Simillar to the Bezier Curve and SPLine its start from a small set of control points. This process is determinated from the starting points and tend towards a smooth line. This consepts is extended to 3D space.

Them are used by adjusting the position of a few points of (a) you control the complex shape of a few control point. 

![[Pasted image 20250328143340.png | 300]]


#### Subdivision classification
###### Primal / Dual
How the new mesh is generated:
- **Primal**: Faces split into sub-faces. In this case  we have two type of vertices, the original and the newer.
- **Dual**: New faces for each vertex, edge face. We add a face for each vertex and edge.

![[Pasted image 20250328143926.png | 350]]

###### Approximating / Interpolating
If the original vertices are preserved:
- **Approximating**: Vertexes on the base mash are just control points.
- **Interpolating**: Vertices of the base  mesh stay fixed and you build a surface interpolating them.

![[Pasted image 20250328144709.png | 450]]
We have also triangles and rectangles classification only for primal type.

### [[Doo-Sabin Algorithms]]

### [[Catmull-Clark Algorithms]]

### [[Loop Schema Algorithms]]

### [[Butterfly Algorithm]]

## Coarsening / Simplification
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

Reduce the number of vertices, minimizing the **approximating error** or keep below the **error** a threshold and minimize the number of vertices.

![[Pasted image 20250403174131.png | 500]]

### Approximating error
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
With this version leads to a different results. This means that the two distance are not symmetric. Approximate as:
1. Sample one surface (uniformly distributed)
2. For each point compute $\max_{y\in S_2}D(x,y)$

This problem is **NP-hard**. It is NP-hard to decide if a given surface of n vertexes can be $\epsilon$-approximated with a surface composed by k vertices.

But even the 2D version of the problem is NP-Hard: Simplifying a polyline to k vertexes so that it $\epsilon$-approximate a optimal simplification using the undirected Hausdorff distance is NP-hard. The same holds when using the directed Hausdorff distance from the input to the output polyline, whereas the reverse can be computed in polynomial time.
### [[Heuristics. Incremental methods]]

### [[Greedy Approach (Boundary Error)]]
# References