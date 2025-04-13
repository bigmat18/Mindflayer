**Data time:** 18:04 - 13-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]]

**Area**: [[Master's degree]]
# Orthoprojection Cuts

This approach does cuts in the following way:
1. We use orthographic Projection from multiple directions. 
2. Map each triangle in the  "best projection". 
3. Use depth peeling for handling overlapping parts

![[Pasted image 20250413180734.png | 600]]

Small isolated pieces are removed and merged with bigger areas, to avoid fragmentation. Useful for Color-to-Geometry mapping. If you have a set of photos aligned over a 3D object they induce a direct parametrization by simply assigning each triangle to the best photo.
### Depth peeling
Deeph peeling is a multi-pass technique to render translucent polygonal geometry without sorting polygons (zbuffer and transparency do not work well together).

The basic idea is to peel geometry from front to back until there is no more geometry to render.

![[Pasted image 20250413181750.png | 500]]

# References