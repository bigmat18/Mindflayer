**Data time:** 17:59 - 27-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Sampling]]

**Area**: [[Master's degree]]
# PDS on Surface

When we work on a surface there is the metrics of distance between two point on a surface.
- **Euclidean distance**: the classic euclidean distance between two points in a 3D space.
- **Geodesic distance**: the quick path on throw the surface to connect two points. For example, on a sphere, if we have two opposite points, the geodesic distance is half circumference. 

![[Pasted image 20250427180638.png | 350]]
The difference in the sampling is that in some case with the euclidean distance if we have an object with a low width, two points in each side can not be sample because the euclidean distance is too low, instead with geodesic it can be.

The [[Hierarchical Dart Throwing (HDT)]] is essentially the same. We need to:
- Replace "[[Uniform Grid|uniform grid]]" with "triagulation"
- Replace "[[Quad-Tree]]" with 1-4 triangle subdivision.
- Replace "Euclidean distance" with "geodesic distance"

![[Pasted image 20250427181544.png | 600]]
1. Pick a point randomly, check that is not closer than r to any other point
2. until the triangle containing the point is not entirely within distance r, iterate 1-4 slitting
3. Remove entirely covered triangles from the activel list. End when the list is empty.

### Cost
**Coverage test**: Constant like in [white07]
**Unbiased sampling**: Logarithmic binning (each bin store a list of triangles within a range area)
1. a = random value in \[0, total_area\] 
2. Linear search on the bins until sum_b > a
3. Pick a triangle and accept with prob triangle_area/bin_area. Repeat until accept.
4. Pick a random point in the triangle

![[Pasted image 20250427183520.png | 300]]


# References
- [white07] K. B. White, D. Cline, and P. K. Egbert,”Poisson disk point sets by hierarchical dart throwing," in Proceedings of the IEEE Symposium on Interactive Ray Tracing 2007.