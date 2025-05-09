**Data time:** 12:29 - 12-10-2024

**Status**: #youngling #note 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Representations]] [[Surface Recostruction]]

**Area**: [[Master's degree]]

# Marching cubes

Marching cubes is an algorithm to convert implicit surface representation to parametric surface. This is a grid based method in which samples the implicit function on a regular grid and process each cell the [[Signed distances field|discrate distance field]] separately.
###### Input 
- A regular 3D grid where each node is asociated with a scalar value $f$ (i.e. a scalar field)
- A scalar value $\alpha$
###### Output
- A surface with scalar value $\alpha$ and non null gradiant (the isosurface)

The value at p is obtained by trilinear interpolation of the values of the vertices of the grid cell containded in:
![[Pasted image 20250508133019.png | 500]]

The general algorithms is:
- For each cell that is intersected by a isosurface S a surface patch is generated
- The collections of all these small pieces eventually yields a triangle mesh approximation of the complete isosurface S.

![[Screenshot 2024-10-12 at 12.48.58.png | 400]]

For each grid edge intersecting the surface S the algorithms compute a sample point approximates this intersection. The sign F differs at the grid edge's endpoints $p_1$ and $p_2$. This approximations is linear along the grid edges. $d_{1} := F(p_1)$ and $d_2 := F(p_2)$ 
$$s = \frac{|d_{2}|}{|d_{1}| + |d_{2}|}p_{1} + \frac{|d_{1}|}{|d_{1}| + |d_{2}|}p_{2}$$
The result sample points of each cell to a loop-up table holding all possibile configurations od edge intersections.

![[Screenshot 2024-10-13 at 15.11.40.png | 400]]

All configurations are $2⁸ = 256$ but only 14 considering rotations, mirroring and complement. For each combinations of field value respect to the threshold, store the corresponding triangulation.

![[Pasted image 20250508134622.png | 400]]

Marching cubes computes intersection points on the edge of a regular grid only, which causes **sharp edges** or **corners**. 
###### Pros
- Quite easy to implement
- Fast and not memory consuming
- Very robust
###### Cons
- **Consistency**: Guarantees a C0 and [[Representing real-world surfaces|manifold]] result: ambiguous cases
- **Correctness**: Return a good approximation of the real surface
- **Mesh complexity**: The number of triangles does not depend on the shape of the isosurface. If we have a plane, we do a lot of triangles in relations to sampling value.
- **Mesh quality**: arbitrarily ugly triangles. If I want an isosurface very close to a 2 vertex aligned on same axes we do a very long and tiny triangle.

### Ambiguous cases
In the following situations:

![[Pasted image 20250506182436.png]]

We don´t know a a surface was connected, it's not enough see the lookup table to fix these types of problems. We need to play with the approximation of the function that we want represents. 

The value of the scalar function inside each cell is interpolated by the (known) value of its 8 corners.
$$T(x,y,z) = axyz + bxy + cyz + dxz + ex + fy + gz + h$$
Where:
- $a = v1 + v3 + v4 + v6 - v0 - v7 - v5 - v2$
- $b = v0 + v2 - v1 - v3$
- $c = v0 + v7 - v4 - v3$
- $d = v0 + v5 - v1 - v4$
- $e = v1 - v0$
- $f = v3 - v0$
- $g = v4$

We evaluate how much worth the field in the interpolated function in the center of cube, and if in this point the value is above the threshold (red) that means we are in first situations (the one at the top) else we are in the second situations
![[Pasted image 20250506183129.png | 200]]

Field value on a cells face:
$$T(0,y,z) = cyz + fy + gz + h$$
$$\frac{\partial T(0,y',z')}{\partial y} = cz'+ f = 0 \Rightarrow z'= -\frac{d}{c} \:\:\:\:\:\: \frac{\partial T(0,y',z')}{\partial z} = cy'+ g = 0 \Rightarrow y'= -\frac{g}{c}$$
![[Pasted image 20250508135626.png | 300]]

We can build an **ELUT: Exhaustive LUT**. For each ambiguous configuration determines the coherent internal triangulation looking at the saddle points

![[Pasted image 20250508135831.png | 350]]

For 3D case is more difficult because we need to evaluete the triliniar above the cube face that we want analyze, like image above.

### Marching Tetrahedral
We can fix ambiguous problem by reduce the complexity of the cells, using **tetrahedral cells** (instead of cubical), this type of marching cube has only 3 configurations ($2⁴ == 16$ permutation of grid values reduce to 3 cases)
- No ambiguities (linear field on a value is planar)
- Boundary cases are easy to be managed too, cases where you get a vertex with exact threshold value
- It may be less correct, if you start from a cubic grid the tetrahedral decomposition is a biasing choice.

![[Pasted image 20250508140629.png | 300]]

To use this type of cells in Marching algorithm cubic cells are partitioned in 5 or 6 tetrahedra. Subdivision determines topology. 
- The 5 decomposition is more simple and obvious, 3 regular tetrahedral and 1 irregular, the bad thing of this decomposition is **oriented**, that means if we take 2 cube and I put one next to the other the diagonal not match. 
- The 6 decomposition is equally simple to build, the diagonal match in this case.

![[Pasted image 20250508141950.png | 150]]

We can also decompose the cube adding one more sample in cubic cell. **Body centered cubic lattice**.
- Unique subdivision
- Equal tetrahedral
- Better surface (better triangles)

To do it we add a point in the center of cube and I decompose with this point.

### Adaptive Triangulation
After we evaluate the righe topology from lookup table we can do a refine step for a better approximation (re-evalueate scalar field). We decompose the surface obtained, we project vertex

![[Pasted image 20250508143509.png | 400]]
### Extended marching cubes
Marching cubes computes intersection points on the edge of a regular grid only, which causes **sharp edges** or **corners**. To resolve this problem there is **Extended marching cubes** algorithm that check the distance function's gradient $\nabla F$ to detect those cells that contain a sharp, and than it find an additional sample point by intersecting the estimated tangent planes at the edge intersection points of the voxel.

![[Screenshot 2024-10-13 at 15.01.29.png | 350]]

By using point and normal information on both sides of the sharp feature, one can find a good estimate for the feature points at the intersection of the tangent elements. 

In the following image, the dashed lines are the result the standard marching cubes algorithm produce, and the bold lines are the tangents used in the extended algorithm.
![[Screenshot 2024-10-12 at 12.54.22.png | 400]]

We have a seguence of steps:
1. We use classic MC to found the intersection points

![[Pasted image 20250508144904.png | 350]]

2. We know the normal on intersections, by this information we can build something that can approssimate the original surface if the normal are different more than theshold.

![[Pasted image 20250508145218.png | 300]]

This make marching cubes for solid but also more difficult to be implemented.
### Dual countering approach
This approach extract meshes from adaptive octrees directly. In contrast to marching cubes this approach generates vertices in the interior of the voxels and constructs a polygon for every voxel edge that intersects the isosurface.

The core idea is, for each vertex generated by marching cube a generate a patch and one quad for each intersected edge (the 4 vertices associated to the patches of the cells sharing the edge). This approach tend to improve triangles quality but introduce complexity. 

A **problem** in this approach yields [[Representing real-world surfaces|non manifolds]] for cell configurations containing multiple surface sheets, an other approach to mitigate this problem is the **cubical marching squares algorithm**.

![[Pasted image 20250508173521.png | 450]]

One of problem of marching cube is that the complexity was independent from the complexity of shape. To fix this problem we can partition the space the [[Quad-Tree|octree]]:
1. Partition the space with an octree.
2. Build the dual grid
3. Run MC on the dual grid (consider non hexahedral cells as HC with collapsed edges)
![[Pasted image 20250508151903.png | 400]]

The main issues is the MC assume that I have equals adjacent cells, to fix this problem if I image that the octree represent a value inside each cells. When we switch from a level to another we image that the level above has the 4 division, we calculate 4 cells and subsequently we "squash" on a side with a linear deformation. 
# References