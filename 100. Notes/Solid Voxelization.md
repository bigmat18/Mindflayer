**Data time:** 00:32 - 08-06-2025

**Status**: #note #youngling 

**Tags:** [[Surface Voxelization and SDF Generation]]

**Area**: [[Master's degree]]
# Solid Voxelization

With a solid voxelization we want describe a volume where each voxel that have the center inside the object will be marked with 1, 0 otherwise. We assume to have a closed and watertight object.
### 4.1 Direct, triangle-parallel Solid Voxelization
We run a thread for each faces of the mesh. We execute for each triangle the following operations:
1. Calculate le **bounding box 2D** in the $yz$ plane. 
2. We execute a double loop that iterate on all $(y,z)$ coordinate inside the bounding box. We image to watch the mesh from $-x$ plane, we see the $yz$ plane where each pixel is a **column of voxel** parallel to $x$ axes. This column is a row of voxel with a $x$ coordinate. We iterate on them.
3. For each column of voxel we check if the center of column of voxel is covered by the triangle.
4. Now we describe the column of voxel like a 3D line:
$$
P(t) = (t, y_{col}, z_{col})
$$
	with $t$ is a parameter long the $x$ axes. We want calculate the intersection with the triangle, we consider only the bounding box (we have already check the belonging to triangle). Describe the bounding box like a surface in 3D space:
$$
	Ax + By + Cz + D= 0
$$
	where $(A, B, C)$ is the normal vector of plane (the normal of triangle $n$), and $D$ is a constant that we can find using one of the 3 triangle vertex, $D = -(n \cdot v_0)$. The intersection is described by:
$$
	n_x(t) + n_y (y_{col}) + n_z(z_{col}) - (n \cdot v_0)= 0
$$
	we should found $t$ that determinate the intersection point long the $x$ axes.
5. We convert $t$ to a voxel with $q = \lfloor t \rfloor$.
6. We employ the **parity principle** to fill the volume. As we traverse a row of voxels, every time we encounter a voxel that marks a boundary, we toggle a fill state using an `xor` operation. When the first boundary voxel is found, all subsequent voxels are flipped to 1. Upon finding the second boundary voxel, the `xor` operation flips them back to 0. This process creates the volumetric effect, ensuring that only the voxels between an odd and an even boundary crossing are ultimately set to 1.

### 4.2 Tile-Based Solid Voxelization
This approach want optimize the previous one, instead to create a thread for each triangle, we create a thread for region (tile). To do that the algorithm is spited in two phases. First we assign triangle for each tile, second we execute the processing.

#### Tile Assignment
We run a thread for each triangle and we calculate at which tile it belong.
###### 1. Calculate the overlapping
###### 2.  Work Queue creation
###### 3. Sorting per tile
###### 4. Compact result


# References