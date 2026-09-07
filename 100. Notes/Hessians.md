---
Data: 2026-03-22T17:28:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Hessians

Because the partial derivative $\frac{\partial f}{\partial x_{i}}: \mathbb{R}^{n} \rightarrow \mathbb{R}$ is itself a function, it has partial derivatives of its own. If we differentiate twice, we obtain the **second-order partial derivative**:
$$\frac{\partial^{2}f}{\partial x_{j}\partial x_{i}} \quad \text{and} \quad \frac{\partial^{2}f}{\partial x_{i}\partial x_{i}} = \frac{\partial^{2}f}{\partial x_{i}^{2}} = [f_{x}^{i}]^{\prime\prime}$$

By computing the [[Jacobian]] of the gradient map $\nabla f(x): \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$, we compute the **Hessian** (matrix) of $f$ at $x$:
$$\nabla^2 f(x) := \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2}(x) & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_n}(x) \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1}(x) & \dots & \frac{\partial^2 f}{\partial x_n^2}(x) \end{bmatrix}$$

For a quadratic function $f(x) = \frac{1}{2}x^{T}Qx + qx$, the Hessian elegantly simplifies to the constant matrix: $\nabla^{2}f(x) = Q$.

Using the Hessian, we can build the **second-order model**, which captures the curvature of the function and acts as a much better local approximation (a multivariate parabola):
$$Q_{x}(z) = L_{x}(z) + \frac{1}{2}(z-x)^{T}\nabla^{2}f(x)(z-x)$$

#### Computational Cost, Symmetry, and the $C^2$ Class
- **Cost:** Calculating and storing the Hessian requires $O(n^2)$ memory (unless the matrix is sparse), making it a massive bottleneck when $n$ is very large (e.g., in Deep Learning).
- **Symmetry (Schwarz's Theorem):** If $\exists \delta > 0$ such that $\forall z \in \mathcal{B}(x,\delta)$ the mixed partials exist and are continuous at $x$, then the Hessian is perfectly symmetric:
  $$\frac{\partial^{2}f}{\partial x_{j}\partial x_{i}}(x) = \frac{\partial^{2}f}{\partial x_{i}\partial x_{j}}(x) \equiv \nabla^{2}f \text{ is symmetric}$$
  A symmetric Hessian mathematically guarantees that all eigenvalues of $\nabla^{2}f(x)$ are real numbers.
- **The $C^2$ Class:** We define $f \in C^{2}$ if $\nabla^{2}f(x)$ is continuous everywhere. This implies a beautiful chain of smoothness: 
  $$\nabla f(x) \in C^{1} \Rightarrow \nabla f(x) \in C^{0} \Rightarrow f(x) \in C^{0}$$
  The $C^2$ class (and strictly speaking $C^3$) is the absolute best scenario for optimization algorithms, though in modern applications it is sometimes necessary to make do with much less.
# References