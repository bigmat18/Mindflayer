---
Data: 2026-02-19T16:37:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Quadratic function

We move from linear functions to the simplest form of quadratic functions. These are functions where variables appear as squares ($x_i^2$) but do not interact with each other (no terms like $x_1 x_2$).

A general quadratic function in \(n\)-dimensions is defined as:
$$
f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x} + c
$$
where:
- $\mathbf{x} \in \mathbb{R}^n$ is the input vector.
- $A \in \mathbb{R}^{n \times n}$ is a symmetric matrix capturing quadratic and cross-terms (e.g., $2h x_i x_j$ for $i \neq j$).
- $\mathbf{b} \in \mathbb{R}^n$ is the linear coefficient vector.
- $c \in \mathbb{R}$ is the constant term.
#### [[Non-Homogeneous (separable)]]
#### [[Homogeneous (general case)]]

#### Optimizing Non-Homogeneous Quadratic Functions
We consider the general quadratic function
$$f(x) = \frac{1}{2}x^T Q x + \langle q, x \rangle$$
- $x \in \mathbb{R}^n$ è il vettore variabile.
- $Q \in \mathbb{R}^{n \times n}$ simmetrica (perché quadratiche reali sono simmetriche).
- $\langle q, x \rangle = q^T x$ è il prodotto scalare (termine lineare non-omogeneo).

To find the optima, we look for points where the gradient vanishes ($\nabla f(x) = Qx + q = 0$). The solvability of this system depends on the matrix $Q$.
##### The Nonsingular Case (Invertible Q)
When $\det(Q) \neq 0$, all eigenvalues are non-zero. The matrix $Q$ is invertible, so the linear system has a unique solution.

We define the stationary point as **$\bar{x} = -Q^{-1}q$**. By substituting $x = z + \bar{x}$ (shifting the coordinate system), the linear term is eliminated, and the function transforms into:
$$f(x) = \frac{1}{2}z^T Q z + c$$
where $c$ is the constant value $f(\bar{x})$.
- **Analysis:** This transformation reduces the problem to the homogeneous case centered at $\bar{x}$. If $Q \succ 0$, $\bar{x}$ is a global minimum; if $Q \prec 0$, it is a global maximum; otherwise, it is a saddle point.
- **Constraints:** While the unconstrained solution is simple, adding box constraints ($x \in [l, u]$) makes the problem NP-hard due to the combinatorial complexity of the boundaries.

##### The Singular Case (Non-invertible Q)
When $\det(Q) = 0$, $Q$ has zero eigenvalues and cannot be inverted. We must analyze the structure using Eigenvalue Decomposition, splitting the space into the **Image** (curved directions) and the **Kernel** (flat directions).

We decompose the linear vector $q$ into two components $q = q_+ + q_0$:
- $q_+$ lies in the Image.
- $q_0$ lies in the Kernel (where $Qq_0 = 0$).

The function transforms to $g(z) = \frac{1}{2}z^T Q z + q_0^T z + c$. The presence of the linear term $q_0^T z$ determines the outcome:

###### Case A: Unbounded (Incompatible)
If **$q_0 \neq 0$**, the vector $q$ has a component "trapped" in the Kernel. The system $Qx = -q$ has **no solution**.
- The function behaves linearly along the Kernel direction.
- Since it is a line on a flat surface, it is unbounded: the minimum is $-\infty$ and the maximum is $+\infty$.

###### Case B: Infinite Solutions (Compatible)
If **$q_0 = 0$**, the vector $q$ lies entirely in the Image. The system $Qx = -q$ is compatible and has **solutions**.
- The linear slope vanishes along the Kernel.
- Instead of a single point, we have **infinite global minima** (assuming convexity) forming a flat affine subspace (like a valley).
- Any solution to the linear system is an optimal point.
# References