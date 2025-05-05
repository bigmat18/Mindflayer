**Data time:** 15:21 - 29-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Differential Geometry]]

**Area**: [[Master's degree]]
# Gradiant, Divergence and Laplacian

## Gradiant

Given a function $F: \mathbb{R}² \to \mathbb{R}$ (our surface) the **gradiant** of F is the vector field $\nabla F : \mathbb{R}² \to \mathbb{R}²$ defined by the partial derivatives:
$$\nabla F(x,y) = \bigg (\frac{\partial F}{\partial x}, \frac{\partial F}{\partial y}\bigg)$$
**Intuitively**: at the point $p_0$ the vector $\nabla F(p_0)$ points in the **direction of greatest change of F**.

![[Pasted image 20250429152020.png | 600]]

## Divergence
Given a function $F(F_1, F_2): \mathbb{R}² \to \mathbb{R}²$ the **divergence** of F is the function $div: \mathbb{R}² \to \mathbb{R}$ defined as:
$$div \: F(x,y) = \partial F_1 / \partial x + \partial F_2 / \partial y$$
**Intuitively**: At the point $p_0$ the divergence $div\: F(p_0)$ is a measure of the extent to which the flow (de)compresses at $p_0$.

## Laplacian
Given a function $F(F_1, F_2): \mathbb{R}² \to \mathbb{R}$ the **Laplacian** of F is the function $\Delta F: \mathbb{R}² \to \mathbb{R}$ defined by the divergence of the gradiant of the partial derivatives.
$$\Delta F = div(\nabla F(x,y)) = \partial² F / \partial x² + \partial² F / \partial y²$$
**Intuitively**: The Laplacian of F at the point $p_0$ measures the extent to which the value of F at $p_0$ differs from average value of F its neighbors.

#### Discrete Differential Operators
We assume that meshes are piecewise linear approximations of smooth surfaces. The approach says to approximate differential properties at point $x$ as spatial average over local mesh neighbourhood $N(x)$ where typically:
- x = mesh vertex
- $N(x)$ = n-ring neighborhood (or local [[PDS on Surface|geodesic]] ball)

Uniform Discretization:
$$\Delta_{uni}f(v) := \frac{1}{|N_1(v)|} \sum_{v_i \in N_1(v)} (f(v_i) - f(v))$$
- Depends only on connectivity, simple and efficient
- Bad approximation for irregular triangulation

**Cotangent formula**: 
$$\Delta_S f(v) := \frac{2}{A(v)} \sum_{v_i \in N_1(v)} (\cot \alpha_i + \cot \beta_i)(f(v_i) - f(v))$$
We need to weight each vertex using the sum of cotangent of the two angles that affect the edge that leads ti the vertex to weight.
![[Pasted image 20250429162908.png | 600]]
This is an interesting thing because this Laplacian is good enough to be used for curvature in discrete cases.

#### Discrete Curvatures
- [[Mean Curvature]]:                  $H = ||\Delta_S x||$
Is the Laplacian calculated with cotangent weight  

- [[Gaussian Curvature]]:            $G = (2\pi - \sum_j \theta_j) / A$
This is difference between $2\pi$ the sum of all angles that affect on vertex.
![[Pasted image 20250429164400.png | 150]]
- Principal Curvatures:           $k_1 = H + \sqrt{H² - G} \:\:\:\:\:\:\:\:\:k_2 = H - \sqrt{H² - G}$


# References