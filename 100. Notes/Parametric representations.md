**Data time:** 13:01 - 12-10-2024

**Status**: #padawan #note 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Representations]]

**Area**: [[Master's degree]]
# Parametric representations

A parametric surface are defined by a vector-valued parameterisation function $f: \Omega \to S$ with $\Omega \subset \mathbb{R}^2$ and $S \subset \mathbb{R}^3$ . For example the **torus** function
$$\begin{cases}x = (R + r \cdot \sin t) \cdot \cos s \\
y = (R + r \sin t) \cdot \sin s \\
z = r \cos t\end{cases}$$
This equation represent the following figure in a 3D space

![[Screenshot 2024-09-29 at 13.05.43.png | 250]]

This representation have the advantage to **reduce many 3D problems on the surface 2D** but, in other hand generating parametric surface parameterisation $f$ can be very complex since the parametric domain $\Omega$ has to match the topology and the matrix surface of the surface $S$, **that means $\Omega$ and $S$ must be similar**.

### Spline surfaces
This is a standard surface representation often used in CAD systems, often called NUMBS, they are used for constructing high-quality surfaces and they can be described by piecewise polynomial or rational B-spline basis function $N_{i}^n(.)$

They are built by connecting several polynomial patches in a smooth $C^{n-1}$ 
$$
(u,v) \to \sum_{i=0}^m \sum_{j=0}^k c_{ij}N_{i}^n(u)N_{j}^n(v)
$$
- $c_{ij} \in \mathbb{R}^3$ is called **control points** or **control mesh** of the spline surface
- $N_i^n(u) >0$ and $\sum_i N_i^n \equiv 1$ and for these each surface point $f(u,v)$ is a convex combination of control points $c_{ij}$ 

![[Screenshot 2024-10-10 at 12.12.06.png | 250]]

A tensor-product surfaces always represents a rectangular domain under the parameterization **f**, if shapes are complex the model has to be decomposed into number of tensor-product patches
# References