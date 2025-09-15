**Data time:** 17:31 - 13-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]] [[Parametrization Techniques]]

**Area**: [[Master's degree]]
# Least Squares Conformal maps

This approach is used to do parametrization using least squares. In this case **does't need the entire boundary to be fixed**. It maintain angle between triangles. Imposing that two vectors on UV maps to 2 orthogonal, same length vectors in 3D.

The idea is the following:
$$
\min_f \sum_{t\in Triangles(i,j,k)} ||A_t \cdot U_t||²
$$
where:
- $A_t$ is a discretizzazione of Cauchy-Riemann condition per $t$
- $U_t$ are the coordinates in 2D (u,v) per triangle vertices
- if $A_t U_T = 0$ is perfect conform, it is not always possibile and for this we use least square

there is also the following formula:
$$
\min_f \sum_{t \in Triangolo} ||J_t(f) - sR_t||²
$$
- $J_t(f)$ is the [[Jacobian Matrix]] of $f$
- s is a scalar
- $R_t$ is a rotation


To resolve least square we impose that Laplace of triangles must have $\sigma_1, \sigma_2$ equals to obtain minimal [[Parametrization Distortion|Angle distortion]]. 

![[Pasted image 20250413174354.png |350]]
In this approach need to fix only 2 vertices to disambiguate.

![[Pasted image 20250413174454.png | 500]]
# References
- [Least Squares Conformal Maps for Automatic Texture Atlas Generation](https://www.cs.jhu.edu/~misha/Fall09/Levy02.pdf)