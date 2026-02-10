---
Data: 2026-02-10T17:59:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Dimensionality Reduction
**Rationale:** N-dimensional data are mapped to 2 or 3 dimensions for better visualization/understanding. Widely used strategy. In general, it is a mapping not a geometric transformation. Different mappings have different properties.
### Principal Component Analysis (PCA)
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

### Locally Linear Embedding (LLE)
LLE attempts to discover nonlinear structure in high dimension by exploiting local linear approximation. We use this reduction when own data are near and manifold with low dimension

![[Pasted image 20260210225313.png]]

- INTUITION: assuming that there is sufficient data (well-sampled manifold) we expect **each data point** and its neighbors **can be approximated by a local linear patch**.
- The patch is represented by a **weighted sum of the local data points**.

To compute local patch we need:
1. Choose a set of data points close to a given one (ball-radius or K-nearest neighbours).
2. Solve $W_{ih}$

![[Pasted image 20260210225902.png | 500]]

We can do a LLE mapping where we define a new function with $Y$ and we need to find $\vec{Y}_i$ which minimizes the embedding cost function:
![[Pasted image 20260210230039.png| 500]]

The LLE algorithms is the following:
1. Compute the neighbors of each data point, $\vec{X}_i$
2. Compute the weights $\vec{W}_{ij}$ that best reconstruct $\vec{X}_i$
3. Compute the vectors $\vec{Y}_i$ that minimizes the cost function.

###### Example
![[Pasted image 20260210230220.png | 500]]

![[Pasted image 20260210230246.png | 350]]

### Isomap
The core idea is to preserve the [[PDS on Surface#PDS on Surface|Geodesic Distance]] between data points. Geodesic is the shortest path between two points on a curved space.

![[Pasted image 20260210230742.png | 250]]

![[Pasted image 20260210230750.png]]

1. **Construct neighborhood graph:** Define graph G over all data points by connecting points $(i,j)$ if and only if the point i is a K neareast neighbor of point j
2. **Compute the shortest path.** Using the Floyd’s algorithm. It is an algorithm to find the shorted paths between all pairs of vertices in a weighted graph
3. **Construct the d-dimensional embedding**

![[Pasted image 20260210231011.png | 550]]

![[Pasted image 20260210231031.png | 550]]


### Autoencoders
Machine learning is becoming ubiquitous in Computer Science. A special type of neural network is called autoencoder. An autoencoder can be **used to perform dimensionality reduction**. 

![[Pasted image 20260210231112.png | 400]]

Multi-layer autoencoder:
![[Pasted image 20260210231136.png | 500]]

### T-SNE
Most techniques for dimensionality reduction **are not able to retain both the local and the global structure of the data in a single map.** This is a usefull to see non linear structures ([[Representing real-world surfaces#Manifoldness|manifold]], cluster ...)  Simple tests on handwritten digits demonstrate this (Song et al. 2007).

Similarities between high- and low- dimensional data points is modeled with c**onditional probabilities**. 
- Conditional probability that the point $x_i$ would peak $x_j$ as its neighbor:

![[Pasted image 20260210231432.png | 450]]

- [[Gaussian Curvature|Gaussian]] centered in $x_i$
- $\sigma_i$ scarto per avere una certa "perplexity" per controllar i vicini effettivi

We are interested only in pairwise distance
$$
p_{i|i} = 0
$$
For the low-dimensional points an analogous conditional probability is used:

![[Pasted image 20260210231928.png]]

- x is the input dimension
- y is the foal dimension

###### Kullback-Leibler Divergence
**Coding theory:** expected number of extra bits required to code samples from the distribution P if the current code is optimize for the distribution Q.

**Bayesian view:** a measure of the information gained when one revises one's beliefs from the prior distribution Q to the posterior distribution P.

It is also called relative entropy.
- Definition for discrete distributions:
![[Pasted image 20260210232055.png | 450]]

- Definition for continuos distributions:
![[Pasted image 20260210232113.png | 450]]

In the SNE (Stochastic Neighbor Embedding) The goal is to minimizes the mismatch between $p_{j|i}$ (x) and $q_{j|i}$ (y). Using the Kullback-Leibler divergence this goal can be achieved by minimizing the function:

![[Pasted image 20260210232226.png | 450]]

Problem of SNE:
- The cost function is difficult to optimize.
- SNE suffers, as other dimensionality reduction techniques, of **the crowding problem**.

The solution in **T-SNE**. SNE is made symmetric: It employs a Student-t distribution instead of
a Gaussian distribution to evaluate the similarity between points in low dimension.

![[Pasted image 20260210232337.png]]

- The crowding problem is alleviated.
- Optimization is made simpler.

# References