**Data time:** 15:27 - 13-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]] [[Parametrization Techniques]]

**Area**: [[Master's degree]]
# How do Parametrization

We have the following elements:
- Triangle mesh $S \subset \mathbb{R}³$
	- Vertices $p_1, \dots, p_{n+b}$
	- Triangles $T_1, \dots, T_m$

- Parameter mesh $\Omega \subset \mathbb{R}²$
	- Parameter points $u_1, \dots, u_{n+b}$
	- Parameter triangles $t_1, \dots, t_m$

- Parametrization function $f: \Omega \to S$
	- Piecewise linear map $f(t_j) = T_j$

![[Pasted image 20250413153444.png | 250]]
In practise the parametrization is another mesh where for each triangle in mesh we interpolate the coordinate of vertices and we optein a points in the parametrization domain.
##### Mesuring Parametrization quality
This is not a easy task to be done in a synthetic way. There are many different measures.

- **Atlas crumbliness and solidity**: Crumbliness is the ratio of the total length of the perimeter of the atlas charts, summed over all charts, over to the perimeter of an ideal circle having the same area as the summed area of all charts.

### [[Mass-Spring Parametrization]]

### [[Harmonic Parametrization]]

### [[Least Squares Conformal maps]]

### [[As-rigid-as-possible Parametrization]]

# References