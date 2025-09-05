**Data time:** 14:40 - 11-08-2025

**Status**: #note #master  

**Tags:** [[3D Geometry Modelling & Processing]] [[Remeshing. Mesh Simplification and Approximation]]

**Area**: [[Master's degree]]
# Variational Shape Approximation
In Variational Shape Approximation (VSA) the input shape is approximated bya a set of proxies, the approximation error is iteratively decreased by clustering faces into best fitting regions.

Let $M$ be a triangle mesh and let $R = \{R_1, \dots, R_k\}$ be a partition of M into k regions, ie $R \subset M$ and we have:
$$
R_1 \cup \dots \cup = M
$$
We also have $P =\{ P_1, \dots, P_2\}$ a set if proxies. A proxy $P_i = (x_i, n_i)$ is a simply a plane in space through the point $x_i$ with normal direction $n_i$.

Define also two metrics that measure a generalized distance of a region $R_i$ to its proxy $P_i$:
$$
L²(R_i, P_i) = \int_{x\in R_i} (n_i^T x - n_i^Tx_i)² dA
$$
the second is base od a measure of the normal field.
$$
L^{2,1}(R_i, P_i) = \int_{x\in R_i} = ||n(x) - n_i||² dA
$$

The goal of VSA is then the following: given a number of K and error metric E (ie either $E=L²$ or $E=L^{2,1}$) find a set $R = \{ R_1, \dots, R_k \}$ and a set $P = \{ P_1, \dots, P_k \}$ of proxies such that the global distortion:
$$
E(R, P) = \sum^k_{i = 1}E(R_i, P_i)
$$
is minimized. Then we can extract a mesh  of the original input from the proxies.

### Algorithm
The VSA iteratively alternates between two phases **geometry partitioning** and **proxy fitting**. At the end we apply  **mesh extraction** phase to get output mesh.

![[Pasted image 20250811163320.png]]
##### Geometry Partitioning
In this phase the algorithm modifies the set R of regions to achieve a lower approximation error while keeping the proxies P fixed.

Given two set of $R = \{R_1, \dots, R_k\}$ and $P = \{P_1, \dots, P_k\}$
```c
// For each region find the seed triangle and initilize a priority queue
for i = 1 to k do:
	select the triangle t in Ri that minimize E(t, Pi)
	Ri = {t}
	set t to conquered
	for all neighbors r of t do:
		insert (r, Pi) into queue
```

In this phase we built a priority queue for each region with seed the best triangle.

```c
// Grow the regions
while the queue is not empty do:
	get (t, Pi) from the queue that minimizes E(t, Pi)
	if t is not conquered then:
		set t to conquered
		Ri = Ri union {t}
		for all neighbors r of t do
			if r is not conquered then
				insert (r, Pi) into queue
```

The algorithm is initialized by randomly picking k triangles $t_1, \dots, t_k$ on the input model, setting $R_i = \{t_i\}$ and initializing $P_i = (x_i, n_i)$ where $x_i$ is an arbitrary point on $t_i$ and $n_i$ is the normal of $t_i$.
##### Proxy Fitting
In this phase the partition R is kept fixed while the proxies $P_i = (x_i, n_i)$ are adjusted in order to minimize $E(R, P)$.

- For the $L²$ metric, the best proxy is the least-squares fitting plane. It can be found using integral principal component analysis
- Instead for $L^{2,1}$ metric the proxy normal $n_i$ i just the area-weighted average of the triangle normals.
##### Mesh Extraction
From an optimal partitioning $R = \{R_1, \dots, R_k\}$ and corresponding proxies $P = \{P_i, \dots, P_k\}$ we can extract a anisotropic remesh.

1. All vertices in the original mesh that are adjacet to three or more different regions are identified.
2. These vertices are projected onto each proxy and their average position in computed. These are called anchor vertices.
3. Anchor vertices are then connected by tracing the boundaries of the region R.
4. The result are triangulated using [[Delaunay Triangulation]] 

# References
- [Variant Shape Approximation by Cohen-Steiner et al](https://www.geometry.caltech.edu/pubs/CAD04.pdf)