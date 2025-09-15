---
Data: 2025-09-08T17:08:00
tags:
  - note
  - master
Connection:
  - "[[3D Geometry Modelling & Processing]]"
  - "[[Differential Geometry]]"
Area: "[[Master's degree]]"
---
# Discrete Differential Operators

We assume that meshes are piecewise linear approximations of smooth surfaces. The approach says to approximate differential properties at point $x$ as spatial average over local mesh neighbourhood $N(x)$ where typically:
- x = mesh vertex
- $N(x)$ = n-ring neighborhood (or local [[PDS on Surface|geodesic]] ball)

#### Uniform Laplacian
This is a uniform discretization of the [[Gradiant, Divergence and Laplacian#Laplacian|Laplacian-beltrami]] operator
$$\Delta_{uni}f(v) := \frac{1}{|N_1(v)|} \sum_{v_i \in N_1(v)} (f(v_i) - f(v))$$
- Depends only on connectivity, simple and efficient
- Bad approximation for irregular triangulation (non uniform mesh) because in some cases we expect a zero value since the mean curvature over the entire region in zero

This formula, applied to the coordinate function x evalutes to the vector pointing from the center vertex $x_i$ to the average of the one-ring vertices $x_j$.
#### Cotangent formula 
A more accurate discretization of the [[Gradiant, Divergence and Laplacian#Laplacian|laplacian-beltrami]] operator can be derived using a mixed finite elment/finit volume method.
$$\Delta_S f(v) := \frac{2}{A(v)} \sum_{v_i \in N_1(v)} (\cot \alpha_i + \cot \beta_i)(f(v_i) - f(v))$$
We need to weight each vertex using the sum of cotangent of the two angles that affect the edge that leads ti the vertex to weight.
![[Pasted image 20250429162908.png | 600]]
This is an interesting thing because this Laplacian is good enough to be used for curvature in discrete cases.

This version has also disaventge:
- the cotangent weight $(\cot \alpha_{i,j} + \cot \beta_{i,j})$ become negative if $\alpha_{i,j} + \beta_{i,j}) > \pi$. This can be lead to flipped triangles in certain application
- the discrete laplacian equation above is not purely **intrinsic**, ie, its evaluation can lead to different results, even for two isometric surfaces with different triangulations
#### Discrete Curvatures
- [[Mean Curvature]]:                  $H = \frac{1}{2}||\Delta_S x||$
Is the Laplacian calculated with cotangent weight  

- [[Gaussian Curvature]]:            $G = (2\pi - \sum_j \theta_j) / A$
This is difference between $2\pi$ the sum of all angles that affect on vertex.
![[Pasted image 20250429164400.png | 150]]
- Principal Curvatures:           $k_1 = H + \sqrt{H² - G} \:\:\:\:\:\:\:\:\:k_2 = H - \sqrt{H² - G}$


# References