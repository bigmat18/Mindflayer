**Data time:** 15:21 - 29-04-2025

**Status**: #note #master

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
**Intuitively**: At the point $p_0$ the divergence $div\: F(p_0)$ is a measure of the extent to which the flow (de)compresses at $p_0$. How much a vectors fiends enter or exit from a point.

## Laplacian
Given a function $F(F_1, F_2): \mathbb{R}² \to \mathbb{R}$ the **Laplacian** of F is the function $\Delta F: \mathbb{R}² \to \mathbb{R}$ defined by the divergence of the gradiant of the partial derivatives.
$$\Delta F = div(\nabla F(x,y)) = \partial² F / \partial x² + \partial² F / \partial y²$$
**Intuitively**: The Laplacian of F at the point $p_0$ measures the extent to which the value of F at $p_0$ differs from average value of F its neighbors.

# References