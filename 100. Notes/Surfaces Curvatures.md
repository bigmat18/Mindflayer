**Data time:** 00:57 - 29-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Differential Geometry]]

**Area**: [[Master's degree]]
# Surfaces Curvatures

We can define **[[Curves#Curvature|curvature]]** for a 2D domain by second derivatives. Define a tangent vector on a parametric surface using and angular form: 
$$t = \cos \phi \frac{x_u}{||x_u||} + \sin \phi \frac{x_v}{||x_v||}$$
in this definition we use $\phi$ to choose a unique direction in this plane.

![[Pasted image 20250429011835.png | 300]]

This concept work very well in a 2D space. In **3D space** we try to reduce the concept of curvature at a 2D domain. To do that consider the plane along n, t and the 2D curve defined on it.

![[Pasted image 20250429012037.png | 300]]

In a formal way, let $t = u_t x_u + v_t x_v$ be a tangent vector at a surface point $p \in S$ represeted as $t = (u_t, v_t)^T$ in parameter space. The **normal curvarure** is the curvature of the planer curve created by intersecting the surface at p with the plane spanned by t and the surface normal:
$$
k_n(\bar{t}) = \frac{\bar{t}^TII\bar{t}}{\bar{t}^T I \bar{t}} = \frac{eu_t^2+2fu_tv_t + gv_t^2}{Eu_t^2 + 2Fu_tv_t + Gv_t^2}
$$
where $II$ denotes the second foundamental form defined as:
$$
II = \begin{bmatrix}e & f \\ f & g\end{bmatrix} = \begin{bmatrix}x_{uu}^Tn & x_{uv}^Tn\\ x_{uv}^Tn &x_{vu}^Tn\end{bmatrix}
$$
### Main curvature directions
- For each direction $t$ we define a curvature value k
- Let's consider the two directions $k_1$ and $k_2$ where the curvature values $k_1$ and $k_2$ are **maximum** and **minimum**

There is a **Euler theorem** that said: $k_1$ and $k_2$ are perpendicular and curvature along a direction t making an angle $\Theta$ with $k_1$ is:
$$k_{\theta} = k_1 \cos² \Theta + k_2 \sin²\Theta$$
This theorem has an important handy impact because it say if in any smooth surface the two interesting curvature are perpendicular each other and define a reference frame very useful.

![[Pasted image 20250429014638.png | 230]]

In a smooth surface for each point, if we use this theorem, we have a direction along the way of max curvature, while the other direction will be along the way that not change a lot.
### [[Gaussian Curvature]]

### [[Mean Curvature]]

# References
