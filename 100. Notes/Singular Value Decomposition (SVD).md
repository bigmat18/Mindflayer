---
Data: 2026-04-06T00:20:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Singular Value Decomposition

### Defining the SVD
**The operational way**: Let $A\in\mathbb{R}^{m\times m}$, and $A^{T}A=V\Lambda V^{T}$ be an eigenvalue decomposition, with $V$ orthogonal. Then, $AV$ satisfies:
$$
(AV)^{T}(AV)=V^{T}A^{T}AV=V^{T}(V\Lambda V^{T})V=\Lambda
$$
This means that the columns of $AV$ are orthogonal, but not orthonormal: the $i$-th column has norm $\sqrt{\lambda_{i}}$. 
**The smart way:** We can scale them by defining $(AV)_{i}=u_{i}\sigma_{i}$ with $\sigma_{i}=\sqrt{\lambda_{i}}$. Then the $u_{i}$ are the columns of an orthogonal matrix $U$.

### Singular value decomposition (SVD)
This gives a variant of the eigenvalue decomposition that is well-defined for every matrix. 

> **Definition** (for square matrices): Each matrix $A\in\mathbb{R}^{m\times m}$ can be decomposed as:
> $$
A = USV^T = 
\begin{bmatrix}
u_1 & u_2 & \dots & u_m
\end{bmatrix}
\begin{bmatrix}
\sigma_1 & & & \\
 & \sigma_2 & & \\
& & \ddots & \\
 & & & \sigma_m
 \end{bmatrix}
 \begin{bmatrix}
 v_1^T \\
 v_2^T \\
 \vdots \\
 v_m^T
 \end{bmatrix}
 $$
> 
>Alternatively, expressed as a **linear combination** of rank-1 matrices:
> $$
 A = u_{1}\sigma_{1}v_{1}^{T}+u_{2}\sigma_{2}v_{2}^{T}+\dots+u_{m}\sigma_{m}v_{m}^{T}
 $$
>with $U, V$ orthogonal and $\sigma_{1}\ge\sigma_{2}\ge\dots\ge\sigma_{m}\ge0$.

**Warning**: In this decomposition $U$ and $V$ are not the inverse of each other! Because of this, we lose the ability to express matrix powers:
$$
A^{2}=A\cdot A=USV^{T}USV^{T}\ne US^{2}V^{T}
$$

### Singular Values vs. Eigenvalues
The $\sigma_{i}$ are called **singular values** and we can take them non-negative and ordered: $\sigma_{1}\ge\sigma_{2}\ge\dots\ge\sigma_{m}\ge0$.
* Singular values $\ne$ eigenvalues. 
* They are always positive and usually more 'spread apart' than the eigenvalues.
* **Uniqueness**: singular values are unique; singular vectors $u_{i}$, $v_{j}$ are not - exactly like eigenvalues / eigenvectors.

### Rectangular matrices
The same theorem holds also for a rectangular matrix, with some changes in the shape of the involved matrices.
Each matrix $A\in\mathbb{R}^{m\times n}$ can be decomposed as $A=USV^{T}$, with $U, V$ orthogonal and $\sigma_{1}\ge\sigma_{2}\ge\dots\ge\sigma_{m}\ge0$. 
Here, $U\in\mathbb{R}^{m\times m}$, $S\in\mathbb{R}^{m\times n}$ (padded with zeros), $V\in\mathbb{R}^{n\times n}$.
###### Example
$$
S=\begin{bmatrix}
\sigma_{1}&0&0&0&0\\ 
0&\sigma_{2}&0&0&0\\ 
0&0&\sigma_{3}&0&0
\end{bmatrix}
$$

$$
A=u_{1}\sigma_{1}v_{1}^{T}+u_{2}\sigma_{2}v_{2}^{T}+\dots+u_{\min(m,n)}\sigma_{\min(m,n)}v_{\min(m,n)}^{T}
$$

### Thin SVD
Note that the sum-of-rank-1 form uses only the first $\min(m, n)$ columns of $U$ and $V$. This suggests a different, more compact form, the **thin** (or economy-sized) SVD.

For **tall-thin matrices**:
$$
A=\begin{bmatrix}U_{0}&U_{c}\end{bmatrix}\begin{bmatrix}S_{0}\\ 0\end{bmatrix}V^{T}=U_{0}S_{0}V^{T}
$$
where $U_{0}\in\mathbb{R}^{m\times n}, S_{0}\in\mathbb{R}^{n\times n}$.

**Computational costs**:
* `[U, S, V] = svd(A, 0)` (thin) costs $O(mn^{2})$ ops for $A\in\mathbb{R}^{m\times n}$ or $A\in\mathbb{R}^{n\times m}$ with $m\ge n$.
* `[U, S, V] = svd(A)` (non-thin) is more expensive: it has to compute and return the large $m\times m$ factor.

### Properties of the SVD: rank, image, kernel
**Rank**: Rank $r =$ number of nonzero singular values: $\sigma_{1}\ge\dots\ge\sigma_{r}>\sigma_{r+1}=\dots=\sigma_{n}=0$.
We can omit row/columns after $r$ in the product:
$$
A = \hat{U}\hat{S}\hat{V}^T = u_{1}\sigma_{1}v_{1}^{T}+u_{2}\sigma_{2}v_{2}^{T}+\dots+u_{r}\sigma_{r}v_{r}^{T}
$$

* **Image** ($Im(A)$): For each $x\in\mathbb{R}^{n}$, $Ax$ is linear combination of $u_{1},\dots,u_{r}$.
* **Kernel** ($Ker(A)$): Any linear combination $y$ of $v_{r+1},\dots,v_{n}$ satisfies $Ay=0$.

# References