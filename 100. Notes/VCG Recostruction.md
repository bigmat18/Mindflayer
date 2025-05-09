**Data time:** 02:39 - 09-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[3D Geometry Modelling & Processing]]
# VCG Recostruction

In main case we have not a point cloud but an [[Range Maps]]. We want to get a nice isosurface from the range maps. We can use an approch similar to [[Mesh Zippering]]. 

This method can be difficult when we have a ton of surfaces on each other. For this reason we decided to switch into a [[Introduction to Surface Reconstruction|volumetric rapresentation]], the basic approch was with surface with [[Normals on 3D Models|normas]] to interpolate.

![[Pasted image 20250509024657.png | 250]]

The basics steps are:
1. Compute [[Signed distances field|signed distance field]] from each range map
![[Pasted image 20250509024759.png | 250]]

2. Average them. This means if in a specific point a surface say the distance is 1 and other say distance 2 we have distance 1,5.
![[Pasted image 20250509024944.png | 250]]

3. Extract the isosurface.

if we do the simple averaging can cause abrupt jumps in the new isosurface.

![[Pasted image 20250509025113.png | 250]]

The solution is weight the averaging by [[PDS on Surface|geodesic distance]] to border

![[Pasted image 20250509025330.png | 250]]


# References