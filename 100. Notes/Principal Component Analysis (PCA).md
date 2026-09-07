---
Data: 2026-07-19T14:52:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Principal Component Analysis (PCA)

- A classic multi-dimensional reduction technique is Principal Component Analysis (PCA).
- It is a linear non-parametric technique.
- The core idea to find a basis formed by the directions that maximize the variance of the data, and reduce the redondance.

The new basis is a lienar combination of the original basis. The PCA founds new axis calls:
1. Linear combination of original features
2. Orthogonal among them
3. Maximize the variance
$$
PX = Y \text{ where } PX = \begin{bmatrix}p_1\\ \vdots \\ p_m\end{bmatrix}\begin{bmatrix}x_1 & \dots & x_n\end{bmatrix}
$$
the part $\begin{bmatrix}x_1 & \dots & x_n\end{bmatrix}$ contains signals + noise after denationalization of matrix axis are dissociated. With this formula next we have:
$$
y_i = \begin{bmatrix}p_1 \cdot x_i \\ \vdots \\ p_m \cdot x_n\end{bmatrix}
$$
Givining a signal with noise:
$$
SNR = \frac{P_{signal}}{P_{noise}}
$$
it can be expressed as:
$$
SNR = \frac{\sigma²_{signal}}{\sigma²_{noise}}
$$
the **numerator** is the variant of the first principal components. the **denominator** is the variant of the last principal components

![[Pasted image 20260210221907.png]]

###### Covariance Matrix
$$
Cov(X) = C_X = \frac{1}{n-1} XX^T
$$
- Square symmetric matrix.
- The diagonal terms are the variance of a particular variable.
- The off-diagonal terms are the covariance between the different variables.

The goals is select the best P. That means:
- **Minimize redundancy** (its equal to maximize SNR)
- **Maximize the variance**

That can be translated into **diagonalize the covariance matrix of Y**.
- High values of the diagonal terms means that the dynamics of the single variables has been maximized.
- Low values of the off-diagonal terms means that the redundancy between variables is minimized.

![[Pasted image 20260210223048.png | 400]]

To solve the PCA we need of **Specturm Theorem**: a symmetric matrix A can be diagonalized by a matrix formed by its eigenvectors as $A = EDE^T$

The column of E are the eigenvectors of A. If we choose $P = E^T$ we can compute the equation above and that maked $C_Y$ diagonalizzabile.

At the end we can compute PCA with the following steps:
1. Organize the data as an m x n matrix.
2. Subtract the corresponding mean to each row. This are the centered data $x_c = x - \mu$
3. Calculate the eigenvalues and eigenvectors of $XX^T$.
4. Organize them to form the matrix P.

The idea to use **PCA for Dimensionality Reduction** is to:
1. find the k-th principal components (k < m). There are the data withe lower appoximation
2. Project the data on these directions and use such data instead of the original ones.
3. This data are the best approximation w.r.t the sum of the squared differences.

If we use only the first k < m components we obtain the best reconstruction in terms of squared error.
![[Pasted image 20260210224824.png | 500]]

PCA as the Projection that Minimizes the Reconstruction Error:
![[Pasted image 20260210224840.png | 500]]

Limits of PCA:
- It is non-parametric -> this is a strength point but it can be also a weak point.
- It fails for non-Gaussian distributed data.
- It can be extended to account for non-linear transformation -> kernel PCA.

![[Pasted image 20260210225006.png | 500]]

### Classic MDS (Multidimensional Data Scaling)
Find the linear mapping $y_i = Mx_i$ which minimizes:
![[Pasted image 20260210225055.png]]

We can use PCA and MDS. We want to minimize $\phi (Y)$  corresponds to maximize:
![[Pasted image 20260210225143.png | 550]]

That is the variance of the low-dimensional points (same goal of the PCA).
- The size of the covariance matrix is proportional to the dimension of the data.
- MDS scales with the number of data points instead of the dimensions of the data (PCA).
- Both PCA and MDS preserve better large pairwise distances.

###### Summoning mapping
Adaptation of MDS by weighting the contribution of each (i,j) pair:

![[Pasted image 20260210231213.png]]

This allows to retain the local structure of the data better than classical multidimensional scaling (the retain of high distances is not privileged).
# References