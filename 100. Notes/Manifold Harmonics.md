---
Data: 2025-09-12T03:33:00
Tags:
  - note
  - youngling
Connection:
  - "[[3D Geometry Modelling & Processing]]"
  - "[[Smoothing]]"
Area: "[[Master's degree]]"
---
# Manifold Harmonics

### Fourier Transformation
the fourier transformation is the classic tool for analysing a signal's frequency spectrum. It maps a univariate function $f: \mathbb{R} \to \mathbb{C}$ from its representation $f(x)$ in teh spatial domain to its representation $F(\omega)$ in the frequency domain:
$$
F(\omega) = \int^{\infty}_{-\infty} f(x)e^{-2\pi i \omega x}dx
$$
$$
f(w) = \int^{\infty}_{-\infty} F(\omega) e^{2\pi i \omega x}d\omega
$$
The fnction $f(x)$ can be considered an element of a certain vector space, which is equipped with the inner product
$$
<f,g> = \int^{\infty}_{-\infty} f(x) \overline{g(x)}dx
$$
where $\overline{(a + ib)} = (a - ib)$ denotes complex conjugation. 

### Manifold Harmonics
The 1D fourier framework is now be generalized to a function $f: S \to \mathbb{R}$ on a (discrete) 2-manifold surface. But it can not be directed translated to a function on manifold.There are the following issues:
- sine and cosine functions
- complex waves $e_w$
are eigenfunctions of an Laplace operator, ie:
$$
\Delta(e^{2\pi i \omega x}) = \frac{d²}{dx²}e^{2\pi i \omega x} = - (2\pi \omega)² e^{2 \pi i \omega x}
$$
The final results with a discretization is:
$$
\Delta f(v_i) = \sum_{v_j \in N_i(v_i)} w_{ij} (f(v_j) - f(v_i))
$$
assume that the weights are not normalized by vertex or Voronoi area. 
# References