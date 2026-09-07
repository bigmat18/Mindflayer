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
>Alternatively, expressed as a **linear combination** of rank-1 matrices:$$A = u_{1}\sigma_{1}v_{1}^{T}+u_{2}\sigma_{2}v_{2}^{T}+\dots+u_{m}\sigma_{m}v_{m}^{T}
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


## Least Squares with the SVD

The Singular Value Decomposition (SVD), specifically the thin SVD, provides a robust method to solve the least-squares problem. The derivation is similar to the one used with the QR decomposition. By substituting the SVD of $A$ ($A = USV^T$) into the least-squares objective function, we get:

$$\vert{}\vert{}Ax-y\vert{}\vert{}=\vert{}\vert{}USV^{T}x-y\vert{}\vert{}=\vert{}\vert{}S\underline{V}^{T}x-U^{T}y\vert{}\vert{}$$

Assuming all singular values $\sigma_i$ are strictly positive (different from 0), the minimum is achieved when we define a vector $z$ whose components are:
$$z_{i}=\frac{u_{i}^{T}y}{\sigma_{i}}$$
From this, we can recover the solution vector $x$:
$$x=Vz=V\Sigma_{0}^{-1}U_{0}^{T}y$$

Putting everything together, the complete explicit formula for the least-squares solution using the SVD is:
$$x=\sum_{i=1}^{n}v_{i}\frac{u_{i}^{T}y}{\sigma_{i}}=V\begin{bmatrix}\frac{1}{\sigma_{1}} & 0 & \dots & 0 \\ 0 & \frac{1}{\sigma_{2}} & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \frac{1}{\sigma_{n}}\end{bmatrix}U^{T}y$$
**Key Observation:** In this solution, the smaller singular values $\sigma_i$ appear in the denominator, meaning they contribute _more_ to the final solution vector $x$ (unless the numerator $u_{i}^{T}y \approx 0$).

##### The Pseudoinverse Formula
The expression above provides a direct way to compute the Moore-Penrose pseudoinverse $A^+$ using only the thin SVD:
$$A^{+}=V\Sigma_{0}^{-1}U_{0}^{T}$$

### Full Rank and the SVD
To guarantee a unique least-squares solution, we need to know when all $\sigma_{i}\ne0$. This is directly tied to the rank of the matrix $A$. Note the relationship with the normal equations matrix $A^T A$:
$$A^{T}A=(USV^{T})^{T}(USV^{T})=VS^{T}SV^{T}=V\begin{bmatrix}\sigma_{1}^{2} & 0 & \dots & 0 \\ 0 & \sigma_{2}^{2} & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \sigma_{n}^{2}\end{bmatrix}V^{T}$$
This factorization shows that the following conditions are mathematically equivalent:
1. $A$ has full column rank.
2. $A^T A$ is invertible.
3. $\sigma_i \neq 0$ for all $i$.

Additionally, the rank of the matrix $r = rank(A)$ is exactly equal to the number of nonzero singular values.

### Rank-Deficient Least-Squares Problems

##### Zero Singular Values
A rank-deficient problem occurs when $r < n$, meaning there are zero singular values: $\sigma_{r+1}=\sigma_{r+2}=\cdot\cdot\cdot=\sigma_{n}=0$.
- In this scenario, $A^{T}A$ is only positive semidefinite, which means the quadratic function is not strongly convex and the minimizer is _not unique_.
- All choices for the components $Z_i$ corresponding to $\sigma_i = 0$ result in valid solutions (minima).

To isolate a single, uniquely determined solution, we typically want the **minimum-norm solution**:
$$x_{*}=arg~min_{x\in arg~min\vert{}\vert{}Ax-y\vert{}\vert{}}\vert{}\vert{}x\vert{}\vert{}$$

This is achieved by taking $z_i = 0$ whenever $\sigma_i = 0$. Essentially, this means replacing $\frac{1}{\sigma_{i}}$ with $0$ in the previous inversion formulas for any $\sigma_j = 0$. The definition of the pseudoinverse $A^+$ is extended to rank-deficient matrices to automatically return this minimum-norm solution: $x_{*} = A^{+}y$.

##### Approximate Dependencies and Perturbations
In real-world applications, exact dependencies (where $\sigma_i = 0$ perfectly) are very rarely encountered. Instead, we often deal with _approximate_ dependencies caused by:
1. **Noise in the data:** Real measurements are rarely perfect.
2. **Inexact computation:** Machine arithmetic often produces $\sigma_n \neq 0$ even when an exact dependency exists, with errors usually on the order of machine precision $u \approx 10^{-16}$.

**Perturbation Theorem:** Let $\sigma_i$ be the singular values of $A$, and $\tilde{\sigma}_{i}$ those of a perturbed matrix $A+E$. Then the difference is bounded by the norm of the error matrix:
$$\vert{}\vert{}\sigma_{i}-\tilde{\sigma}_{i}\vert{}\vert{}\le\vert{}\vert{}E\vert{}\vert{}$$

_Note:_ When analyzing tiny singular values, computing the SVD is numerically more accurate than computing the eigenvalues (`eig`) of $A^T A$, because eigenvalues square the condition number and are more susceptible to precision loss.

### Handling Small Singular Values
##### Truncated SVD
Many real-world matrices feature decaying singular values. Small $\sigma_i$ values act as massive amplifiers in the denominator, causing the exact solution $x$ to vary wildly and become highly unstable. However, the most meaningful features in many applications (like image compression or eigenfaces) correspond to the _large_ singular values.

To obtain a better solution from an application standpoint, we can deliberately ignore the contribution of the small singular values. This is called the **Truncated SVD**:
$$x_{reg}=\sum_{i=1}^{k}v_{i}\frac{u_{i}^{T}y}{\sigma_{i}},$$

(for a strategically chosen $k < r$). While $x_{reg}$ is no longer the exact mathematical minimizer of $\vert{}\vert{}Ax-y\vert{}\vert{}$, it frequently yields vastly superior practical results by filtering out noise-amplified components.

##### [[Tikhonov Regularization (Ridge Regression)]]
An alternative solution to the issue of tiny singular values is to change the problem entirely. Instead of just minimizing the residual, we introduce a **penalty term** to discourage solutions with a large norm:

$$min_{x\in\mathbb{R}^{n}}\vert{}\vert{}Ax-y\vert{}\vert{}^{2}+\alpha^{2}\vert{}\vert{}x\vert{}\vert{}^{2}$$

(for a chosen scalar $\alpha > 0$). We can cleverly rewrite this new objective function as a standard least-squares problem of an augmented system:
$$\vert{}\vert{}Ax-y\vert{}\vert{}^{2}+\alpha^{2}\vert{}\vert{}x\vert{}\vert{}^{2}=\bigg\vert{}\bigg\vert{}\begin{bmatrix}A\\ \alpha I\end{bmatrix}x-\begin{bmatrix}y\\ 0\end{bmatrix}\bigg\vert{}\bigg\vert{}^{2}.$$

Thanks to this block-matrix expression, we can derive the explicit solution formula for Ridge Regression:
$$x_{\alpha}=\begin{bmatrix}A\\ \alpha I\end{bmatrix}^{+}\begin{bmatrix}y\\ 0\end{bmatrix}=(A^{T}A+\alpha^{2}I)^{-1}A^{T}y.$$

The addition of $\alpha^2 I$ guarantees that the augmented matrix has full column rank for any $\alpha > 0$, avoiding singularity.
Using the SVD of $A$, the Tikhonov solution can be written as:
$$x_{\alpha}=\sum_{i=1}^{n}v_{i}\frac{\sigma_{i}}{\sigma_{i}^{2}+\alpha^{2}}u_{i}^{T}y.$$

**Filter Factor Analysis:** The function $f(\sigma)=\frac{\sigma}{\sigma^{2}+\alpha^{2}}$ acts as a smooth version of the Truncated SVD:
- When $\sigma\gg\alpha$, $f(\sigma)\approx\frac{1}{\sigma}$, behaving similarly to the true Least Squares solution.
- When $\sigma\ll\alpha$, $f(\sigma)\approx\frac{\sigma}{\alpha}\approx0$, effectively ignoring the small singular values without a harsh cutoff.

# References