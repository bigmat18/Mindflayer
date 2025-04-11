**Data time:** 16:50 - 11-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]]

**Area**: [[Master's degree]]
# Parametrization Distortion

The concepts of distortions is essential to determinate the quality of a parametrization.
- **Angle preservation**: Conformal. In this case we want preservate angles.
![[Pasted image 20250410122421.png | 150]]
- **Area preservation**: Equiareal, In this case instead we want preserve areas.
![[Pasted image 20250410122446.png | 150]]
- **Area and Angle**: Isometric. This conserve perfectly the image between the domains. This happens when the surface create a **null gaussian curvature**.
![[Pasted image 20250410122515.png | 150]]
### Calculate distortion
What happens to the surface point $f(u,v)$ as we more a tiny little bit away from (u,v) in the parameter domain? We approximate with first order Taylor expansion. We have $f_u = \frac{\partial f}{\partial u}$ and $f_v = \frac{\partial f}{\partial v}$
$$\tilde{f}(u + \Delta u, v + \Delta v) = f(u,v) + f_u(u,v) \Delta u + f_v(u,v) \Delta v$$
$$\tilde{f}(u + \Delta u, v + \Delta v) = p + J_f (u) \begin{bmatrix}\Delta u \\ \Delta v\end{bmatrix} \:\:\:\:\:\: J_f = U\Sigma V^T = U \begin{bmatrix}\sigma_1 & 0 \\ 0 & \sigma_2 \\ \sigma & 0\end{bmatrix} V^T$$
$J_f$ is the Jacobian of $f$, ie the 3x2 matrix with partial derivatives of $f$ as column vectors. To describe the Jacobian we have $U$ and $V^T$ that describe the rotation, while $\Sigma$ encodes how much the two axies that we have use to do the evaluation are stretches.

![[Pasted image 20250410124709.png | 400]]

If we consider singular value decomposition (SVD) of the Jacobian: singular values $\sigma_1 \geq \sigma_2 \geq 0$ and **ortonormale** matrices $U \in \mathbb{R}^{3\times 3}$ and $V \in \mathbb{R}^{2 \times 2}$.
- The transformation $V^T$ first rotates all points around $U$ such that the vectors $V_1$ and $V_2$ are in alignment with the $U-$axes and the $V$-axes afterwards 
- The transformations $\Sigma$ than **streches** by the factor $\sigma_1$ in the $u$-direction and by $\sigma_2$ in the $v$-direction
- The transformations $U$ finally maps the unit vectors $(1,0)$ and $(0,1)$ to vectors $U_1$ and $U_2$ in the tangent plane $T_p$ at p.

In practice the values $\sigma_1, \sigma_2$ describe the amount of the local deformations.
- $\sigma_1 = \sigma_2 = 1$ preserves areas, angles and lengths.
![[Pasted image 20250411171839.png | 400]]
- $\sigma_1 / \sigma_2 = 1$ we preserves angles
![[Pasted image 20250411172041.png | 400]]
- $\sigma_1 \cdot \sigma_2 = 1$ we preserves areas.
![[Pasted image 20250411172203.png | 400]]


# References