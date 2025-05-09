**Data time:** 14:12 - 09-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Poisson Surface Reconstruction

We reconstruct the surface of the model by solving for the **indicator function** of the shape. The indicator function is 1 inside an object and 0 outside:
$$\chi_M(p) = \begin{cases}1 & p\in M\\0&p\notin M\end{cases}$$
![[Pasted image 20250509141613.png | 150]]

My problem is go from a sets of oriented points to indicator function.

![[Pasted image 20250509141732.png | 350]]
To do that we pass through indicator gradient, because there is a relationship between the normal field and gradient of indicator function, the gradient is oriented in same direction of [[Normals on 3D Models|normal]] of points.

![[Pasted image 20250509142030.png | 400]]

The core idea is represents the normal set by a vector field $\vec{V}$ and we try to find the function $\chi$ whose gradient best approximate $\vec{V}$.
$$\min_{\chi}||\nabla\chi - \vec{V}||$$
This is a differential equations that became a Poisson problem. Applying the divergence operator, we can transform this into a Poisson problem:
$$\nabla\cdot(\nabla \chi) = \nabla \cdot \vec{V} \Leftrightarrow \Delta \chi = \nabla \cdot \vec{V}$$

### Implementation
The algorithm usually use [[Quad-Tree|adapted quadtree]]. Given the points the septs are:
1. Set oct-tree
![[Pasted image 20250509144312.png | 200]]

2. Compute vector field
	2.1 Define a function space
	![[Pasted image 20250509144429.png | 250]]
	
	2.2 Splat the samples
	![[Pasted image 20250509144545.png | 200]]

	For each cells of oct-tree we sum the normals of point cloud for the point in cell. We optain a adaptive vector field. From this we have defined the vector field
$$\sum_{s\in S}|P_s|\tilde{F}_{s\cdot p} s.\vec{N} dp \equiv \vec{V}$$
	![[Pasted image 20250509144922.png | 200]]

3. Compute indicator function. The indicator function has the gradient aligned with the defined points 
	3.1 Compute divergence
	![[Pasted image 20250509145253.png | 200]]
	
	3.2 Solve Poisson equation
	![[Pasted image 20250509145411.png]]


4. Extract iso-surface.

This algorithm is the stato of arts. It's very easy to [[Introduction to parallel and distributed systems|parallelized]]
# References