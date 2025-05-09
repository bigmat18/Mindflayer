**Data time:** 03:09 - 09-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Radial Basis Functions (RBF)

This method was a solutions for the [[Metaballs]] method, that follow the general schema:
$$f(x) = p(x) + \sum_i \omega_i \varphi (||x - x_i||) \:\:\:\:f(x_i) = f_i$$
Weights are $\omega_i \in \mathbb{R}$, RBF is a function $\varphi: \mathbb{R} \to \mathbb{R}$, p is a polynome.

![[Pasted image 20250509031400.png |  500]]

In sintesi they are a set of method that compute linear system where the complexity of the linear system is in order to the number of samples. For that reason is scale bad above 1000 points, because the resolution of the linear system became too complex in term of complexity. To fix that we use [[Bounded RBD]] method.
# References