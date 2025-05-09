**Data time:** 15:41 - 07-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Mesh Zippering

The **input** is a triangulated [[Range Maps |ranges maps]] (not just point clouds). Work is pairs:

1. Remove overlapping portions. Maintain the minimal overlapping
![[Pasted image 20250507155241.png | 500]]
2. Clip one RM against the other. With the edge of second mesh cut alla triangle of first mesh.

![[Pasted image 20250507155338.png | 400]]

3. Remove small triangles.

![[Pasted image 20250507155424.png | 300]]

This algorithm is not so trivial to implement and there are many small cases where strange things happens and we need to handle. For example: **remove overlapping regions** is a complex problem. Moreover hole may appear, to be fixer later.

![[Pasted image 20250507155750.png | 300]]


# References