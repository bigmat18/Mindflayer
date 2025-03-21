**Data time:** 13:01 - 12-10-2024

**Status**: #padawan #note 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Representations]]

**Area**: [[Master's degree]]
# Implicit representations

A surface represented with implicit representation (or volumetric) A surface defined where the points of the 3D space satisfy a certain property (usually given function = 0)
$$S = \{p \in \mathbb{R} : f(p) = 0\}$$
![[Screenshot 2024-10-08 at 12.54.11.png | 400]]


In this representation we have a $F: \mathbb{R}^3 \to \mathbb{R}$ where, by conventions, **negative function value design points outside the object and positive inside**. With these properties we can make complex objects by Boolean operations applied to geometric primitives.
![[Screenshot 2024-10-08 at 13.33.04.png | 500]]
- Implicit surfaces can be deformed by increasing or decreasing the function value.
- Any scalar multiple $\lambda F$ yields the same zero-set.

###### Signed distance function
Most common and naturale implicit representation, it maps each 3D points x to its signed distance d(x) from the surface S. The absolute value $|d(x)|$ measures the distance of x to $S$. The sign determined by whether or not x is in the interior of $S$.
###### Regular grid
In order to efficiently process implicit surfaces representations F is typically discretised in some **bounding box around the object use a succulently dense grid with nodes $g_{ijk} \in \mathbb{R}^3$**. The most common representation is a uniform grid of samples value $F_{ijk} := F(g_{ijk})$. This representation grows cubically it the precision is increased by reducing the edge length of grid voxels. 
# References