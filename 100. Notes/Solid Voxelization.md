**Data time:** 00:32 - 08-06-2025

**Status**: #note #youngling 

**Tags:** [[Solid Voxelization and SDF Generation]] [[Parallel Surface and Solid Voxelization on GPUs]]

**Area**: [[Master's degree]]
# Solid Voxelization

With a solid voxelization we want describe a volume where each voxel that have the center inside the object will be marked with 1, 0 otherwise. We assume to have a closed and watertight object. Recall that solid voxelization essentially boils down to rasterizing the object into a multi-sliced frame buffer, where a fragment effects flipping the inside/outside state of all voxels below. 

### 4.1 Direct, triangle-parallel Solid Voxelization
Similar to the surface voxelization approaches, we parallelized over all triangles, dedicating one thread per triangle. We execute for each triangle the following operations:
##### Algorithm
###### 1. Calculate Voxel inside BB
Fist we calculate le **bounding box 2D** in the $yz$ plane for the triangle. After that we execute a double loop that iterate on all $(y,z)$ coordinate inside the bounding box. We image to watch the mesh from $-x$ plane, we see the $yz$ plane where each pixel is a **column of voxel** parallel to $x$ axes. This column is a row of voxel with a $x$ coordinate. We iterate on them.
###### 2. Test center of voxel-triangle intersection
For each column of voxel we check if the center of column of voxel is covered by the triangle. To do that we use the [[Surface Voxelization|edge function decribed in second condition in surface voxelization section]]. If the test is passed we move on to the next step.
###### 3. Found x value
In this part we what found the "depth" of intersection between the column of voxel and the triangle, in particular we want obtain the $x$ value. To perform that we describe the column of voxel like a 3D line:
$$
P(t) = (t, y_{col}, z_{col})
$$
with $t$ is a parameter long the $x$ axes. We want calculate the intersection with the triangle, we consider only the bounding box (we have already check the belonging to triangle). Describe the bounding box like a surface in 3D space:
$$
	Ax + By + Cz + D= 0
$$
where $(A, B, C)$ is the normal vector of plane (the [[Normals on 3D Models|normal]] of triangle $n$), and $D$ is a constant that we can find using one of the 3 triangle vertex, $D = -(n \cdot v_0)$. The intersection is described by:
$$
	n_x(t) + n_y (y_{col}) + n_z(z_{col}) - (n \cdot v_0)= 0
$$
we should found $t$ that determinate the intersection point long the $x$ axes. To calculate the normal of the the triangle, we can do the following operations:
$$
	edge_1 = v_1 - v_0 \:\:\:\:\:\:edge_2 = v_2 - v_0
$$
and after we perform che cross product between the two edge $n = edge_1 \times edge_2$ and we found A, B and C, and with them also D. We convert $t$ to a voxel with $q = \lfloor t \rfloor$, this is the value of depth.

###### 4. Flip operations (XOR)
We employ the **parity principle** to fill the volume. As we traverse a row of voxels, every time we encounter a voxel that marks a boundary, we toggle a fill state using an `xor` operation. When the first boundary voxel is found, all subsequent voxels are flipped to 1. 

Upon finding the second boundary voxel, the `xor` operation flips them back to 0. This process creates the volumetric effect, ensuring that only the voxels between an odd and an even boundary crossing are ultimately set to 1.

##### Issues
This approach has **three main weakness**:
1. The number of relevant voxel columns can vary significantly among the triangles processed by one warp, leading to under-utilization.
2. For large voxel grids, flipping all voxels in the derived x range poses high memory [[Processing Bandwidth|bandwidth requirements]].
3. Many [[Atomic Instructions|atomic operations]] may have to be sentimentalized, negatively affecting performance. 


### 4.2 Tile-Based Solid Voxelization
This approach want optimize the previous one, instead to create a thread for each triangle, we create a thread for region (tile) . To do that the algorithm is spited in two phases. First we assign triangle for each tile, second we execute the processing. 
#### Tile Assignment
We run a thread for each triangle and we calculate at which tile it belong.
###### 1. Calculate the overlapping
First for each triangle (assigned to a thread) we execute the overlap test like section 4.1 but not for voxel but for each tile inside bonding box (like previous example). When the test has success we increment a counter assign to the triangle. The output is an array (length equals to number of triangles) where for each position we have the number of tiled overlapped by triangle.
- **Input**: Triangles Buffer of size n.
- **Output**: Buffer of integer where, for each position we have the number of tiled overlapped by triangle.
###### 2.  Exclusive Scan
In this second phase we execute a scan using a library. Scan operation create a new buffer with same length of input buffer, but for each position we store the value of cumulative sum of previous elements in the array like:
```
Input:  [3, 0, 2, 4]
Output: [0, 3, 3, 5]
```
in own case we will know for each triangle at which start to insert the couple (tile, triangle) without raise write conflicts. Moreover we can get the size of queue that will contains the pairs. It can be calculate summing the last element of input with last element of output -1.
```
in_buffer[N-1] + out_buffer[N-1] - 1 = final_size
```
We use [CUB](https://nvidia.github.io/cccl/cub/) library to execute this calculation in a efficient way, in particular we use the `cub::DeviceScan::ExclusiveSum` function.

###### 3. Work Queue Population
Now we run again a kernel with a thread for each triangle and re-execute the tile intersection test, but this time we put the value `(tile_index, triangle_index)` in the buffer allocated by size obtained in previous step. To calculate position we use the results of scan operations:
```
work_queue[scan_buffer[triangle_index] + k-th_tile_found] = (tile_index, triangle_index)
```
- **Input**: `scan_buffer` and empty `work_queue` with size N
- **Output**: the filled `work_queue` in the following way: `[(tile_j, triangle_0), (tile_k, triangle_0), (tile_l, triangle_1) ...]` (work queue is sorted per triangle)

###### 4. Sorting per tile
From previous step we obtain a buffer of pairs sorted by triangle, but we want a buffer sorted by tile. To achieve this we perform a [[Radix Sort]] using the `tile_index`. We use the **[CUB](cub::DeviceRadixSort::SortPairs)** library again with `cub::DeviceRadixSort::SortPairs`.
- **Input**: `work_queue` sorted by triangle
- **Output**: `work_queue` sorted by tile
###### 5. Compact result
What we have prove step number for is the `work_queue` array sorted by tiled, it can look as following:
```
id | tile_id | triangle_id  
-------------------------------  
0  |    5    | 120  
1  |    5    | 345  
2  |    8    | 99  
3  |    8    | 120  
4  |    8    | 411  
5  |    8    | 732    
6  |    17   | 99
...
```

This queue is not sufficient to execute the tiled operations, this because we don't know which tiles are active (how many tiles), and for each tile where the lists of triangles start. To fix these problems we execute a **compact operations**. 
1. We allocate a buffer of 0 and 1 called `flag_buffer`
2. We iterate the `work_queue` and we a tile_id value changes we mark this position with 1.
```
work_queue.tile_id: [ 5, 5, 8, 8, 8, 8, 17, ...]  
flag_buffer:        [ 1, 0, 1, 0, 0, 0, 1, ...]
```
3. We execute a pack operations where (for example in a [[CUDA Kernels]]) we run a thread for each position of `flag_buffer`. If the value is equals to 1 we copy the data on index i from `work_queue` to two new arrays
```
active_tiles_list:
id | tile_id  
-----------------  
0  | 5  
1  | 8  
2  | 17

tile_offsets_buffer:
id | start_index  
---------------------  
0  | 0  
1  | 2  
2  | 6
```

To do this operation without reinvent the wheel we use ttwo functions of CUB:
- `cub::DeviceSelect::Unique` this take in input an ordered array and return a compact versione. In own case the output is `active_tiles_list`

- `cub::DeviceRunLengthEncode::Encode` this take in input an ordered array (own `work_queue` after sorting) and return two array, the second one is a list of length of each "block" in ordered input array. The second return value can be used to obtein own `tile_offsets_buffer`, we need to applay `cub::DeviceScan::ExclusiveSum` on it.

#### Tile Processing
In this section we execute the exact computation to voxelize a mesh. We run a [[Logical & Physical View of Threads-Warps#Logical & Physical View of Threads-Warps|warp]] (32 threads) for each active tile. A tile is 4x4 grid of voxel, thus we have 16 voxel, we can run 2 threads per voxel, each thread per voxel perform operation with half of triangle assign to the tile.
###### 1. Loading data on [[Shared Memory (SMEM) on GPU|Shared Memory]]
In this first part we use the power of shared memory to obtimize the access to the memory for threads in a single warp. We allocare a shared memory with a fixed size (in this paper it is 14), and the first N threads perform a copy from global to shared of triangles data. 

If the number of data is largest than the `BATCH_SIZE` decided at begning we execute a loop. In this loop we perfom this loading operations in a sequeze of triangles chunck and after, for each iteration, we execute in the same way the operation 2 and 3.
###### 2. Loop on each triangle and test it
In this operation we loop throw the triangles loaded in shared memory and we perform the [[Surface Voxelization#Second condition|second condion]] testing if the center of the voxel is inside triangle and after we calculate the depth to execute the flip. All this operations are descibed in the section above [[Solid Voxelization#4.1 Direct, triangle-parallel Solid Voxelization|direct, traingle-parallel]]  version.
###### 3. Execute [[Solid Voxelization#4. Flip operations (XOR)|flip operation]]
Now like in section [[Solid Voxelization#4.1 Direct, triangle-parallel Solid Voxelization|4.1]] we execute the flip operation, however we don't need the atomic operations because one thread at time work on a voxel row. This improve a lot the performance. Moreover we can adopt the same optimization above, and execute the XOR operation one time for "word" (in the paper 32 bit). This reduce the access of global memory,

# References
- [Fast Parallel Surface and Solid Voxelization on GPUs di Michael Schwarz e Hans-Peter Seidel (2010)](https://michael-schwarz.com/research/publ/files/vox-siga10.pd)