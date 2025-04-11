**Data time:** 11:55 - 10-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]]

**Area**: 
# Introduction to Parametrization

To parameterize a surface we need a strategy to flatten a 3D surface on 2D domain, introducing a few distortion as possibile. Also we need a strategy to introduce cuts.
## Flattening a Surface
We define a **surface** like $S \subset \mathbb{R}^3$, and a parameter domain $\Omega \subset \mathbb{R}^2$. Mapping using $f: \Omega \to S$ and $f^{-1}: S \to \Omega$

![[Pasted image 20250410120250.png | 500]]

### Ortho Projection

![[Pasted image 20250410120529.png | 500]]
$$S = \{(x,y,z) \in \mathbb{R}^3 > x^2 + y^2 + z^2 = 1, z \geq 0\} \:\:\:\:\:\: \Omega = \{(u,v) \in \mathbb{R}^2 : u^2 + v^2 \leq 1\}$$
$$f^{-1}(x,y,z) = (x,y) \:\:\:\:\:\:\: f(x,y) = (x,y, \sqrt{1 - u^2 - v^2})$$
### Cylindrical Coords

![[Pasted image 20250410121428.png | 500]]
$$S = \{(x,y,z) \in \mathbb{R}^3 : x^2 + y^2 = 1, z \in [0,1]\} \:\:\:\:\:\: \Omega = \{(\phi, h) \in \mathbb{R}^2 : \phi \in [0, 2\pi), h \in [0, 1]\} \:\:\: f(\phi, h) = (\sin \phi, \cos\phi, h)$$
## Minimize Distortion 
- **Angle preservation**: Conformal
![[Pasted image 20250410122421.png | 150]]
- **Area preservation**: Equiareal
![[Pasted image 20250410122446.png | 150]]
- **Area and Angle**: Isometric
![[Pasted image 20250410122515.png | 150]]
### Distortion
What happens to the surface point $f(u,v)$ as we more a tiny little bit away from (u,v) in the parameter domain? We approximate with first order Taylor expansion. We have $f_u = \frac{\partial f}{\partial u}$ and $f_v = \frac{\partial f}{\partial v}$
$$\tilde{f}(u + \Delta u, v + \Delta v) = f(u,v) + f_u(u,v) \Delta u + f_v(u,v) \Delta v$$
$$\tilde{f}(u + \Delta u, v + \Delta v) = p + J_f (u) \begin{bmatrix}\Delta u \\ \Delta v\end{bmatrix} \:\:\:\:\:\: J_f = U\Sigma V^T = U \begin{bmatrix}\sigma_1 & 0 \\ 0 & \sigma_2 \\ \sigma & 0\end{bmatrix} V^T$$
$J_f$ is the Jacobian of $f$, ie the 3x2 matrix with partial derivatives of $f$ as column vectors.

![[Pasted image 20250410124709.png | 400]]


# References
