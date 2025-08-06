**Data time:** 13:38 - 07-06-2025

**Status**: #note #youngling #paper

**Tags:** [[CSG on Mesh using Voxelization and SDF]]

**Area**: [[Master's degree]]
# Parallel Surface Voxelization
This paper presents data-parallel algorithms for surface and solid voxelization on graphics hardware. First, a novel conservative surface voxelization technique, setting all voxels overlapped by a mesh’s triangles, is introduced, which is up to one order of magnitude faster than previous solutions leveraging the standard rasterization pipeline. 

![[Pasted image 20250621234707.png | 300]]

We then show how the involved new triangle/box overlap test can be adapted to yield a 6-separating surface voxelization, which is thinner but still connected and gap-free. Complementing these algorithms, both a triangle-parallel and a tile-based technique for solid voxelization are subsequently presented

### 3. [[Surface Voxelization]]
In this section we describe how to check if a voxel is part of triangle, in other world which voxel are intersected by a face. We use a **conservative approach** where a voxel will be actives if a triangle touch it (merely touched by a triangle).

### 4. [[Solid Voxelization]]
With a solid voxelization we want describe a volume where each voxel that have the center inside the object will be marked with 1, 0 otherwise. We assume to have a closed and watertight object. Recall that solid voxelization essentially boils down to rasterizing the object into a multi-sliced frame buffer, where a fragment effects flipping the inside/outside state of all voxels below. 

# References
- [Fast Parallel Surface and Solid Voxelization on GPUs di Michael Schwarz e Hans-Peter Seidel (2010)](https://michael-schwarz.com/research/publ/files/vox-siga10.pd)
- [Fast 3D triangle-box overlap testing (SAT)][https://dl.acm.org/doi/abs/10.1080/10867651.2001.10487535]