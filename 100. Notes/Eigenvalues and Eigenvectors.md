---
Data: 2025-11-20T12:43:00
Tags:
  - note
  - youngling
Connection:
  - "[[Linear Algebra]]"
  - "[[Computational mathematics for learning and data analysis]]"
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

#### Quadratic form
For a fixed symmetric matrix $Q=Q^T$, consider $f(x)=x^TQx$. In the **geometric idea** it's like paraboloids.

**Theorem**: Let Q be a symmetric matrix with minimum and maximum eigenvalue $\lambda_{min}, \lambda_{max}$. For any vector $x\in \mathbb{R}^n$, we have:
$$
\lambda_{min}||x||^2 \leq x^T Q x \leq \lambda_{max}||x||^2
$$
**Proof**: First an easy case: $Q=\wedge$ diagonal

![[Screenshot 2025-11-20 at 13.51.08.png | 500]]
If we replace all $\lambda_i$ with $\lambda_{min}$ it gets smaller, and vice vers. In the general case:
$$
x^TQx = x^T(U\wedge U^T)x = c^T \wedge c
$$
for $c=U^Tx$ with $||c|| = ||x||$

#### Positive definiteness
Note: given $x, c=U^Tx = U^{-1}x$ is the vector of its **coordinates** in the basis of eigenvectors $U$

**Theorem**: 
$$\lambda_{min}||x||^2 \leq x^TQx \leq \lambda_{max}||x||^2 \text{ or } \lambda_{min} \leq \frac{x^TQx}{||x||^2} \leq \lambda_{max}$$
or alternatively
$$
\lambda_{min} \leq u^T Qu \leq \lambda_{max} \text{ for each } u \text{ with } ||u||=1
$$
in particular:
- if $\lambda_i \geq 0$ for each eigenvalue of Q, then $x^TQx \geq 0$ for each vector $x$ ($Q$ is called **positive semidefinite**, $Q \succeq 0$)
- if $\lambda_i > 0$ for each eigenvalue of Q, then $x^TQx > 0$ for each vector $x \neq 0$ ($Q$ is called **positive definite**, $Q \succeq 0$)

Moreover, these are 'if and only if'.

###### Properties of $A^TA$
For any $A \in \mathbb{m\times n}$ (possibly rectangular), $A^TA$ is a valid product and gives a square, symmetric matrix
- $A^TA$ is positive semidefinite: because $x^TA^TAx = ||Ax||^2 \geq 0$
- The same properties hold also for $AA^T$

#### Complex matrices
Most of these properties work also for matrices with complex entries, with one change: **replace each $A^T$ with $\overline{A^T}$** (traspose + entrywise conjugate). Often denoted with $A^*$ or $A^H$
$$
||x||^2_2 = x^*x = \overline{x_1}x_2 + \overline{x_2}x_2 + \dots + \overline{x_m}x_m = |x_1|^2 + \dots + |x_m|^2
$$
with is always real $\geq 0$. Some terminology changes:
- $UU^* = Id$: **unitary** matrix
- $Q=Q^*$: **Hermitian** matrix 
# References