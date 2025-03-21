**Data time:** 12:29 - 12-10-2024

**Status**: #youngling #note 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Representations]]

**Area**: [[Master's degree]]

# Marching cubes

Marching cubes is an algorithm to convert implicit surface representation to parametric surface. This is a grid based method in which samples the implicit function on a regular grid and process each cell the [[Signed distances field|discrate distance field]] separately.

- For each cell that is intersected by a isosurface S a surface patch is generated
- The collections of all these small pieces eventually yields a triangle mesh approximation of the complete isosurface S.

![[Screenshot 2024-10-12 at 12.48.58.png | 400]]

For each grid edge intersecting the surface S the algorithms compute a sample point approximates this intersection. The sign F differs at the grid edge's endpoints $p_1$ and $p_2$. This approximations is linear along the grid edges. $d_{1} := F(p_1)$ and $d_2 := F(p_2)$ 
$$s = \frac{|d_{2}|}{|d_{1}| + |d_{2}|}p_{1} + \frac{|d_{1}|}{|d_{1}| + |d_{2}|}p_{2}$$
The result sample points of each cell to a loop-up table holding all possibile configurations od edge intersections.

![[Screenshot 2024-10-13 at 15.11.40.png | 400]]


Marching cubes computes intersection points on the edge of a regular grid only, which causes **sharp edges** or **corners**. 

### Extended marching cubes
To resolve this problem there is **Extended marching cubes** algorithm that check the distance function's gradient $\nabla F$ to detect those cells that contain a sharp, and than it find an additional sample point by intersecting the estimated tangent planes at the edge intersection points of the voxel.

![[Screenshot 2024-10-13 at 15.01.29.png | 350]]

By using point and normal information on both sides of the sharp feature, one can find a good estimate for the feature points at the intersection of the tangent elements. 

In the following image, the dashed lines are the result the standard marching cubes algorithm produce, and the bold lines are the tangents used in the extended algorithm.
![[Screenshot 2024-10-12 at 12.54.22.png | 400]]
### Dual countering approach
This approach extract meshes from adaptive octrees directly. In contrast to marching cubes this approach generates vertices in the interior of the voxels and constructs a polygon for every voxel edge that intersects the isosurface.

A **problem** in this approach yields [[Representing real-world surfaces|non manifolds]] for cell configurations containing multiple surface sheets, an other approach to mitigate this problem is the **cubical marching squares algorithm**.

# References