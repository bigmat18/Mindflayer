---
Data: 2026-07-19T15:21:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Matrix Norms

### Matrix Norms

Recall: $\vert{}\vert{}v\vert{}\vert{}_{2}=\sqrt{v^{\top}v},$ and $\vert{}\vert{}Uv\vert{}\vert{}_{2}=\vert{}\vert{}v\vert{}\vert{}_{2}$ for orthogonal U. Just as we measure the "length" of a vector, one can define a norm for matrices, too.

##### Definition (induced matrix norm)
Given a norm on vectors (e.g., $\vert{}\vert{}\cdot\vert{}\vert{}_{2},\vert{}\vert{}\cdot\vert{}\vert{}_{\infty},...)$, we can define a corresponding norm on matrices:

$$\vert{}\vert{}A\vert{}\vert{}:=max_{v\ne0}\frac{\vert{}\vert{}Av\vert{}\vert{}}{\vert{}\vert{}v\vert{}\vert{}}=max_{\vert{}\vert{}u\vert{}\vert{}=1}\vert{}\vert{}Au\vert{}\vert{}.$$

**Idea:** it's the smallest value of || A|| that ensures $\vert{}\vert{}Av\vert{}\vert{}\le\vert{}\vert{}A\vert{}\vert{}\vert{}\vert{}v\vert{}\vert{}$ for all v. In other words, an induced matrix norm measures the maximum stretching or amplification that the matrix $A$ applies to any vector $v$. This general construction works for every vector norm $(\vert{}\vert{}\cdot\vert{}\vert{}_{1},\vert{}\vert{}\cdot\vert{}\vert{}_{2},\vert{}\vert{}\cdot\vert{}\vert{}_{\infty}...)$.

**Properties** For each choice of matrices A, B and vector v for which the operations make sense, a valid matrix norm must satisfy the following:

1. **Positivity:** $\vert{}\vert{}A\vert{}\vert{}\ge0$, with equality iff A is all-zeros;
2. **Homogeneity:** $\vert{}\vert{}\alpha A\vert{}\vert{}=\vert{}\alpha\vert{}\vert{}\vert{}A\vert{}\vert{}$ for each $\alpha\in\mathbb{R}$;
3. **Triangle Inequality:** $\vert{}\vert{}A+B\vert{}\vert{}\le\vert{}\vert{}A\vert{}\vert{}+\vert{}\vert{}B\vert{}\vert{}$;
4. **Sub-multiplicativity:** $\vert{}\vert{}AB\vert{}\vert{}\le\vert{}\vert{}A\vert{}\vert{}\vert{}\vert{}B\vert{}\vert{}$;
5. **Compatibility with vector norm:** $\vert{}\vert{}Av\vert{}\vert{}\le\vert{}\vert{}A\vert{}\vert{}\vert{}\vert{}v\vert{}\vert{}$ (if same norm for matrices and vectors).
    

### Our favorite norm
Our favorite norm: $\vert{}\vert{}A\vert{}\vert{}_{2}$. It satisfies the invariant property under orthogonal transformations: $\vert{}\vert{}A\vert{}\vert{}_{2}=\vert{}\vert{}AU\vert{}\vert{}_{2}=\vert{}\vert{}UA\vert{}\vert{}_{2}$ for each orthogonal U. (People often omit the subscript 2.)

### Frobenius norm
Other matrix norm of a different kind: Frobenius norm. Instead of being induced by a vector norm measuring the maximum stretching, it treats the matrix as one large vector and calculates the square root of the sum of the absolute squares of its elements:

$$\vert{}\vert{}A\vert{}\vert{}_F = \sqrt{a_{11}^2 + a_{12}^2 + \dots + a_{mn}^2}$$

It satisfies all the properties in the previous slide (reducing to $\vert{}\vert{}v\vert{}\vert{}_{F}=\vert{}\vert{}v\vert{}\vert{}_{2}$ on vectors); in particular, $\vert{}AU\vert{}\vert{}_{F}=\vert{}\vert{}UA\vert{}\vert{}_{F}=\vert{}\vert{}A\vert{}\vert{}_{F}.$

However, it does not come from the 'induced' construction.

### Norm and SVD
There is a fundamental connection between matrix norms and Singular Value Decomposition (SVD).
Since orthogonal matrices do not change $\vert{}\vert{}\cdot\vert{}\vert{}_{2}$, we have:

$$\vert{}\vert{}A\vert{}\vert{}_{2}=\vert{}\vert{}USV^{T}\vert{}\vert{}_{2}=\vert{}\vert{}S\vert{}\vert{}_{2}=\sigma_{1}$$

(Why is $\vert{}\vert{}S\vert{}\vert{}_{2}=\sigma_{1}$ for the diagonal matrix S in SVD? By a similar argument to the one we used for $\lambda_{min}x^{T}x\le x^{T}Ax\le\lambda_{max}x^{T}x.)$ This means the L2 norm of a matrix is exactly its largest singular value.

Similarly, for the Frobenius norm, the squared norm is the sum of the squared singular values:

$$\vert{}\vert{}A\vert{}\vert{}_{F}^{2}=\sum_{i=1}^{min(m,n)}\sigma_{i}^{2}.$$

### Eckart-Young theorem

> **Theorem**: For a matrix A with SVD $A=USV^{T}$, the solution of$$min_{rankX\le k}\vert{}\vert{}A-X\vert{}\vert{}$$for both $\vert{}\vert{}\cdot\vert{}\vert{}_{2}$ and $\vert{}\vert{}\cdot\vert{}\vert{}_{F}$ is given by truncated SVD:
> $$X = u_{1}\sigma_{1}v_{1}^{T}+u_{2}\sigma_{2}v_{2}^{T}+\cdot\cdot\cdot+u_{k}\sigma_{k}v_{k}^{T}$$

This theorem states that if you want to find the best low-rank approximation of a matrix (i.e., simplifying it while losing the least amount of information according to the L2 or Frobenius norms), you simply construct a new matrix by keeping only the top $k$ largest singular values and their corresponding singular vectors.

# References