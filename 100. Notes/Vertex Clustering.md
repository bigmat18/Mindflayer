**Data time:** 17:34 - 10-08-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Mesh Simplification and Approximation]]

**Area**: [[Master's degree]]
# Vertex Clustering

Usually very efficient and robust. Computation typically linear in the **number of vertices**. However the quality of resulting meshes is not always satisfactory.

The basic idea is: for a given approximation tolerance $\epsilon$ 
1. we partition the bouning box space arount the object into cells with diameter smaller thane the tollerance
2. For each cell:
	1. Compute representative vertex position with we assign to all vertices that tall into this cell
3. For each face:
	- If two or three of their corners lie in the same cell it degenerate
	- otherwise it not degenerate
4. For each pair of cluster P and Q
	- If for $p \in P$ and $q \in Q$  vertices in cluster exists edge (p,q) therefore exists edge $(p_c, q_c)$ with $p_c, q_c$ the representative vertex in cluster

![[Pasted image 20250810180157.png | 450]]
###### Topological changes
Topological changes occur when a part of a surface that collapse into a single point is no [[Representing real-world surfaces#Manifoldness|2-manifold]] and it happens when is not homemorphic to a disk, ie, when two different sheets of the surface pass throgu a single $\epsilon$-cell.

This can be and advantage because the scheme is able to change the topology of the given model, and we can effectively reduce the object complexity.
###### Computational Efficiency
It is determined by the effor it takes to map the mesh vertices to clusters. For simple uniform spatial grids this can be achived in linear time with small constrants.

### Computing Cluster Representatives 
Exists differents way to compute the center of each cells. The average is one of them, it is very simple but usually not enough. A more reasonable choise is based on finding the optimal vertex position as least-square approsimation.

Consider a triangle $t_i$ within the current cell of intereset. Les us denote by $P_i = (x_i, n_i)$ with $x_i$ an arbitrary vertex on the plane and $n_i$ the unit normal vector of $t_i$. With $d_i = n_i^T x_i$ the squared distance of a point x from the plane $P_i$ ca nbe computes as:
$$
dist² (x_i, P_i) = (n_i^Tx - d_i)²
$$
The sum of the quadratic distance to the supportin planes $P_i$ of all triangle $t_i$ within a cell C is given by:
$$
E(x) = \sum_{t_i \in C} \bar{x}^T Q_i \bar{x}
$$
the resulint error is measure is called the **quadratic error metric (QEM)**. The optima lposition x minimizing the quadrati error can be computed as the solution of the least square system:
$$
\big( \sum_i n_i n_i^T \big) x = \big( \sum_i n_i d_i \big)
$$
whic can be obatined from the matrix Q as
$$
Q = \begin{bmatrix} A & -b \\ -b^T & c \end{bmatrix}
$$
# References