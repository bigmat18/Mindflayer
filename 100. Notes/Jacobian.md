---
Data: 2026-03-22T17:27:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Jacobian

When dealing with vector-valued functions $f: \mathbb{R}^n \to \mathbb{R}^m$, where $f(x) = [f_1(x), f_2(x), \dots, f_m(x)]$, the partial derivative handles the extra index seamlessly:
$$\frac{\partial f_{j}}{\partial x_{i}}(x) = \lim_{t\rightarrow0} \frac{f_{j}(x_{1},...,x_{i-1},x_{i}+t,x_{i+1},...,x_{n})-f_{j}(x)}{t}$$

Grouping all $m \times n$ partial derivatives gives us the **Jacobian** matrix. It is an $m \times n$ matrix with the transposed gradients of each scalar component $f_j$ acting as its rows:
$$Jf(x) := \begin{bmatrix} \nabla f_1(x)^{T} \\ \nabla f_2(x)^{T} \\ \vdots \\ \nabla f_m(x)^{T} \end{bmatrix}$$
# References