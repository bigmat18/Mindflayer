**Data time:** 15:00 - 10-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Smoothing]]

**Area**: [[Master's degree]]
# Toubin Smoothing

This is the main approach to fix the problem of shrinking in [[Laplacian Smooth]]. With toubin smoothing we try to approach the problem like a signal-processing problem, and for each steps we don't move the vertex to the average but instead we do two steps:

1. Compute the laplacian displacement for each vertex and moves the vertices by $\lambda$ times this displacement.
2. Then compute again the laplacian and moves back each vertex by $\mu$ times the displacement.

where $\lambda > 0, \mu > 0$ are two constants. In other worlds we try to do two [[Laplacian Smooth|diffusion flow]], one to a direction for a certain frequency and the other in other direction for another frequency.

![[Pasted image 20250510151004.png | 500]]


# References