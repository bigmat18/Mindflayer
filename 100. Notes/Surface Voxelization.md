**Data time:** 00:32 - 08-06-2025

**Status**: #note #youngling 

**Tags:** [[CSG on Mesh using Voxelization and SDF]] [[Parallel Surface and Solid Voxelization on GPUs]]

**Area**: [[Master's degree]]
# Surface Voxelization

In this section we describe how to check if a voxel is part of triangle, in other world which voxel are intersected by a face. We use a **conservative approach** where a voxel will be actives if a triangle touch it (merely touched by a triangle).
#### 3.1 Triangle/box overlap test
The test is based on two condition that must be true if a voxel wont to be marked 1. Given a triangle $T$ with vertices $(v_0, v_1, v_2)$ and axis-aligned box (voxel) $B$
1. $T$ plane overlaps $B$
2. for each of the three coordinate planes $(xy, xz, yz)$ the 2D projection of $T$ and $B$ into this plane overlap.
##### First condition
To calculate this condition we use the idea of **signed distance** from a point to a plane. In our case the plane is the triangle, the point will be one of the 8 vertices of voxel. if $f(q)$ is the function that descrive the plane with $q$ is the point of voxel we have:
- $f(q) > 0$ the point is on **positive** side, where the normal is oriented
- $f(q) < 0$ the point is on **negative** side, where the normal is oriented
- $f(q) = 0$ the point is exactly on the surface.

If 2 of 8 vertex have different signed, or 1 have 0 we consider the voxel. To optimize this process we consider only two vertex $v_{min}$ and $v_{max}$. To found them we consider the plane equation $f(x) = n_xx + n_yy + n_z z + D$. A property of linear function is: the minimum and maximum value on a convex domain (like a voxel) are always the angles vertices. In practise we can use the max value and min value of x,y voxel to calculate this condition.

For `min_x, min_y, min_z` and `max_x, max_y, max_z` we use the size of voxel and the position in the grid. To decide which value chose we see the $n$ vector:
- **To build $v_{max}$**
	- X: if $n_x >0$ we take `max_x`, otherwise `min_x`
	- Y: if $n_y >0$ we take `max_y`, otherwise `min_y`
	- Z: if $n_z >0$ we take `max_z`, otherwise `min_z`
- **To build $v_{min}$**
	- X: if $n_x >0$ we take `min_x`, otherwise `max_x`
	- Y: if $n_y >0$ we take `min_y`, otherwise `max_y`
	- Z: if $n_z >0$ we take `min_z`, otherwise `max_z`

Net we apply the test and calculate:
$$
dist_{min} = f(v_{min}) \:\:\:\:\:dist_{max} = f(v_{max})
$$
if $dist_{min}, dist_{max}$ are equal or one of two is equal to 0 the plane intersects the voxel, or:
$$
dist_{min} \cdot dist_{max} \leq 0
$$

##### Second condition
This condition prevent the false positive, they can happen because in the first condition we consider a plane, that is infinite to definition, with this condition we limite the value only at intersection inside the finite bounding box. 

First we must define the **Edge function**. Each edge of a triangle can be see like a infinite line, we can use the line equation like edge function. it is defined in the following way:
$$
E(x,y) = ax + by + c = 0
$$
where $a= y_0 - y_1$, $b = x_1 - x_0$ and $c=-(ax_0 + by_0)$ for the edge composed by $v_0 = (x_0, y_0)$ and $v_1 = (x_0, y_1)$. The calculation of $a, b, c$ comes from the calculation of the normal vector of edge. This formula are correct if we do the assumption that the vertex are ordered so that the normal (a, b) points inwards. To generalise for all cases we have 
$$
n_e = (a, b) = (-e_{y}, e_{x}) =(y_0 - y_1, x_1 - x_0) \cdot \begin{cases}1 & n_z \geq \\ -1 & n_z < 0\end{cases}
$$
where $n_z$ is the $z$ value of the triangle normal and $e = v_1 - v_2 = (e_x, e_y)$. This can be generalize with all plane (XY, XZ, YZ) changing the normal value and the coordinate.

Now we consider a projection of the voxel and the triangle in 2D. We must test the intersection between  2D box and 2D triangle.
1. We check intersection between bounding box of triangle and 2D voxel. This can be done with a simple **AABB intersections**.
2. We check if doesn't a separation line between the two elements. We use the lines defined the edge functions, **SAT** method.

For second points we use the **SAT** method, but with a simplification. The original SAT says if two convex objects are not overlapped with each other iff exists a line (called "separating axis") where the projection of the two objects not overlapped. For our case we should test the normal of edge (3 for triangle, 2 for quad). This paper use an optimization of SAT, it use the edge of triangles as potential separating axis, if the square is fully on the one edge external half-plane, there is not overlapping. To avoid tests for each of 4 vertex we choose the vertex with the high priority to be inside.

To found this value we must found the value that maximize $E(x,y)$, it depends only from the signs of $a,b$. For example for the XY plane:
- For X coordinate: 
	- $a>0$ we take the maximum value of voxel `max_x`
	- $a<0$ we take the minimum value of voxel `min_x`
- For Y coordinate:
	- $b>0$ we take the maximum value of voxel `max_y`
	- $b<0$ we take the minimum value of voxel `min_y`
with this value $P=(P_x, P_y)$ we calculate $E(P_x, P_y) = e$ and checks:
- If $e<0$ XY test fails
- else $e >= 0$ we must check the other edge.
if all edges tests don't fails and AABB test is true we concluded that there is overlapping on XY plane. This test must be done for XY, XZ and YZ if all three tests are true and condition a is ok we can check the voxel with 1.

# References
- [Fast Parallel Surface and Solid Voxelization on GPUs di Michael Schwarz e Hans-Peter Seidel (2010)](https://michael-schwarz.com/research/publ/files/vox-siga10.pdf)
- [Fast 3D triangle-box overlap testing (SAT)](https://dl.acm.org/doi/abs/10.1080/10867651.2001.10487535)