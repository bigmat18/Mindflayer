**Data time:** 17:07 - 29-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Differential Geometry]]

**Area**: [[Master's degree]]
# Curvature via Surface Fitting

A lot of time the curvature with [[Gaussian Curvature]] has problems based on triangulations. Many time in real situations to calculate curvature we do fitting. We take a raidius r of the neighborhood of each point p is used as a scale parameter:
1. Gather all faces in a local neighborhood of radius r
2. Set an axis $w =\frac{1}{n}\sum^n_{i=1}n_j$ where $n_v$ is the number of vertices gathered and $n_i$ is the surface normal at each such vertex
3. Discard all vertices $v_i$ such that $n_i \cdot w < 0$
4. Set a local frame (u,v,w) where u and v are any two orthogonal unit vectors lying on the plane orthogonal to w, and such that the frame is right-handed
5. Express all vertices of the nighborhood in such a local frame with origin at p
6. Fit ti these points a polynomial of degree two through p ([[Least Squares Conformal maps|least square]] fittig)

$$f(u,v) = au² + bv² + cuv + du + eu$$
Curvatures at p are computed **analytically** via first and second fundamental forms of f at the origin.

An important aspects is the choice of r. It can be used to remove noise and it allow to have curvatures extracted at different scales.

![[Pasted image 20250429172825.png | 600]]
# References