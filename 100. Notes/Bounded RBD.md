**Data time:** 03:18 - 09-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Bounded RBD

Evolution of [[Radial Basis Functions (RBF)]] where we use **bounded** radial basis functions. A radial basis function is a function that aproximate a surface and it was built with a sum of basis where the domain is all the set of points. A **bounded RBF** is the same of normal radial basis function but the functions have a limited domain.
$$\varphi(d) = \begin{cases}(1-d)^pP(d)&d < 1\\0 & d\geq1\end{cases} \:\:\:\:\:P(d)= polynome\:\:with\:\:degree\:\:6$$
The value of f is determined only locally (withing the radius 1). Use $\varphi\left( \frac{d}{R} \right)$ to adapt to the point cloud resolution. The resulting matrix is **sparse**, the fitting is local.

![[Pasted image 20250509032605.png | 250]]


# References