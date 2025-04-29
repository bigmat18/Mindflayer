**Data time:** 14:49 - 26-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Spatial indexing]]

**Area**: [[Master's degree]]
# Quad-Tree

The plane is recursively subdivided in 4 sub-regions (equal to each other) by couple of orthogonal planes. We can't memorize anything to describe cells, we need only the content of cells

![[Pasted image 20250426145507.png | 400]]
they are widely used for terrain rendering: each cross in the quadtree is associated with a height value. In a planar are we don't need informations and for areas with high we divided into cells.

![[Pasted image 20250426145744.png | 400]]

### Oct-tree (3D)
It's the same as quad-tree but in 3 dimensions. We divide a volume in 8 parts. Extraction of isosurfaces on large dataset:
- Build an octree on the 3D dataset
- Each nodes store min and max value of the scalar field
- When commuting the isosurface for alpha, nodes whose interval doesn't contain alpha are discarded.

![[Pasted image 20250426150143.png | 300]]

### Advantages of quand/oct tree
Position and size of the cells are implicit, they can be explored without pointers bu using linear array (convenient only if the hierarchies are comple). Where:

**Quadtree**
$$Children(i) = 4i + 1, \dots, 4 \cdot (i + 1) \:\:\:\: Parent(i) = \lfloor i / 4 \rfloor$$
**Octree**
$$Children(i) = 8i + 1, \dots, 8 \cdot (i + 1) \:\:\:\: Parent(i) = \lfloor i / 8 \rfloor$$

We can store the structures only with an **reordering of points structure**. Beyond octree and quadtree there is the structure base on **Z-ordering**.

# Reference