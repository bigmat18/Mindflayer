**Data time:** 17:07 - 22-10-2024

**Status**: #note #youngling 

**Tags:** [[Surface Representations]] [[3D Geometry Modelling & Processing]]

**Area**: [[Master's degree]]
# Surface Conversions

Each representations are usually finite samplings, for example triangle mesh for parametric surface or uniform/adaptive grind in implicit case. These conversions corresponds to a re-sampling step.
#### Implicit to parametric
Form a implicit (or volumetric) representations to a triangle mash (parametric) is called **isosurfaces extraction**. The de-facto algorithm used for it is **[[Marching cubes]]**. An alternative su marching cubes is a **3D Delaunay triangulation** 
#### Parametric to implicit
The conversion from parametric surface to implicit can be done very efficiently by **voxelization** or **3D scan-conversions** but the result is piecewise constant, this because a surface is not smooth everywhere and a piecewise linear or trilinear approximation seems to be the best compromise between accuracy and efficiency.

For a polygonal meshes the conversion to an implicit requires the computation of **[[Signed Distances Field (SDF)]]** to the triangle. To do that, we need to:
- Found efficiently triangle using spatial data structures, for examples **KD-tree**
- Using an algorithm to compute efficiently the distance of entity grid, it's called **Fast marching**
# References