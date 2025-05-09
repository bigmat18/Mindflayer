**Data time:** 13:28 - 09-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Moving Least Square Reconstruction

In this approach we use moving Least Square (like [[Least Squares Conformal maps]]). In moving LS we try to rebuild a fitting in a limit domain weighted on a set of samples, this allow to do a sets of fitting with a weight that is a [[Variabili Aleatorie Notevoli|Gaussian]] on my domain.

- **Least Square (LS)**:
![[Pasted image 20250509135329.png | 450]]

- **Weighted Least Square (WLS)**:
![[Pasted image 20250509135419.png | 420]]

- **Moving Least Square (MLS)**:
![[Pasted image 20250509135524.png | 300]]

Our approach is **iterative**: project the points near the surface onto the surface
1. $\min_{n,t}\sum^N_{i=1} \langle n, p_i - r -tn \rangle² \theta(||p_i - r - tn||)$
2. $\min_{n,t}\sum^N_{i=1} (g(x_i, y_i) - f)² \theta(||p_i - q||)$
3. Move $r$ to $q + g(0,0)n$

We try to do a fitting of polynomial, we start from a set of points and try to build the best fitting plane and after use this plane to build a parametrization express the other points like distance from the plane.

We can use the projection of a point along the normal on polynomial to move the sample into a more smooth position, with this method we can move the point cloud to a smooth representations of surface.

![[Pasted image 20250509141227.png | 450]]
# References