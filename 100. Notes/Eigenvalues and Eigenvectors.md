---
Data: 2025-11-20T12:43:00
Tags:
  - note
  - youngling
Connection:
  - "[[Linear Algebra]]"
  - co
Area: "[[Master's degree]]"
---
# Eigenvalues and Eigenvectors

Given a **square** matrix $A\in \mathbb{R}^{m\times m}$ if $Av = v\lambda$ for $\lambda \in \mathbb{R}$ and $v\in \mathbb{R}^m$ then we call $\lambda$ **eigenvalue** and $v$ **eigenvector** of A. Remember from linear algebra: almost all atrices $A$ can be written as:

![[Screenshot 2025-11-20 at 12.45.30.png | 550]]
Here $w_i^T$ = rows of $V^{-1}$.

**Geometric Idea**: in a suitable basis, $A$ is diagonal.

Behavior under repeated application of matrix:
![[Screenshot 2025-11-20 at 12.47.28.png | 500]]

**Theorem**: in general, for any polynomial $p(x) = p_0 + p_1x + p_2x^2 + \dots + p_dx^n$
$$
p(A) = p_0Id + p_1 A +\dots + p_dA^d = V 
\begin{bmatrix}
p(\lambda_1)\\
&p(\lambda_2)\\
&&\ddots\\
&&&&p(\lambda_m)
\end{bmatrix}
V^{-1}
$$
- **Eigenvalue** are well-defined for each matrix (up to reordering)
- **Eigenvectors** are not:
	- if $v$ is an eigenvector, any multiple $\alpha v$ is, too
	- if $v, w$ are two eigenvectors with **the same** (repeated) eigenvalue $\lambda$, then any linear combination $\alpha v + \beta w$ is, too.
	- Extreme case: any vector is an eigenvector of the identity matrix $Id = VIdV^{-1}$ for any invertible $V$

Some matrices have only **complex** eigenvalues: $\begin{bmatrix}2&7\\-3&2\end{bmatrix}$
Some do not have enough eigenvectors to form a basis: $\begin{bmatrix}1&1\\0&1\end{bmatrix}$
Neat result: for **symmetric matrices**, nothing goes wrong.

**Theorem**: if $A=A^T$ we can **always** find $U, \wedge$ s.t. $A=U\wedge U^{-1}$. Moreover eigenvalues $\lambda_i$ are **all real**, and **we can choose** $U$ [[Orthogonality|orthogonal]]

# References