---
Data: 2026-07-19T15:21:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# QR Factorization

### Introduction to QR Factorization

There is a different algorithm to solve least-squares problems based on a different matrix factorization known as the **QR factorization**. While it is not as powerful or revealing as the Singular Value Decomposition (SVD), it is significantly easier to compute. **Core Idea:** It mixes the concept of Gaussian elimination and LU factorization with orthogonal transformations.

### The Case of a Vector

**Problem:** Given a vector $x\in\mathbb{R}^{n}$, the goal is to find an orthogonal matrix $Q$ such that multiplying it by $x$ zeroes out all components except the first:

$$Qx = \begin{bmatrix} s \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} = se_{1}$$

_(Note: $e_1$ is defined as the first column of the identity matrix $I$)_. **Remark:** Since orthogonal matrices preserve the Euclidean norm (length), the scalar $s$ can only be $s = \pm\vert{}\vert{}x\vert{}\vert{}$.

### Householder Reflectors
A Householder reflector is an orthogonal matrix used to perform a mathematical reflection. **Lemma:** For every vector $v\in\mathbb{R}^{m}$, the matrix defined as:

$$H = I - \frac{2}{v^{T}v}vv^{T}$$

is both orthogonal and symmetric. It can also be written equivalently as $I-\frac{2}{\vert{}\vert{}v\vert{}\vert{}^{2}}vv^{T}$ or $I-2uu^{T}$, where $u=\frac{1}{\vert{}\vert{}v\vert{}\vert{}}v$ is the normalized vector with a norm of 1. **Proof of Properties:** We can verify directly that $HH^{T}=I$ (which proves it is orthogonal) and $H=H^{T}$ (which proves it is symmetric). **Geometric Idea:** These matrices act as reflections (mirroring) with respect to the plane perpendicular to the vector $v$.

##### Cost-Saving Trick
By rearranging parentheses, we can apply the transformation much more efficiently. For any vector $x\in\mathbb{R}^{m\times m}$, we can compute the product $Hx$:

$$Hx = (I - 2uu^{T})x = x - 2u(u^{T}x)$$

This calculation takes $\mathcal{O}(m)$ operations, and applying $HA$ to any matrix $A\in\mathbb{R}^{m\times m}$ takes $\mathcal{O}(m^{2})$ operations.

##### Transforming Vectors by Reflecting
**Lemma:** Let $x$ and $y$ be two vectors such that $\vert{}\vert{}x\vert{}\vert{}=\vert{}\vert{}y\vert{}\vert{}$. If one chooses the vector $v = x - y$, then the Householder matrix $H = I - \frac{2}{v^{T}v}vv^{T}$ guarantees that $Hx = y$. **Geometric Idea:** Reflecting through the plane perpendicular to the difference $x - y$ directly sends $x$ into $y$. In the context of the QR algorithm, we want to zero out the lower elements of a column, so we specifically take $y = \vert{}\vert{}x\vert{}\vert{}e_{1}$.

### Numerical Stability in Implementation
A basic implementation of finding the Householder vector might look like this:

```
function [u, s] = householder_vector(x)
    s = norm(x);
    v = x;
    v(1) = v(1) - s;
    u = v / norm(v);
```

**Reason for Instability:** If the original vector $x$ is very close to a multiple of $e_1$ (for example, $x_1$ is huge compared to the rest), subtracting two almost-equal values ($x_1$ and $s$) causes an issue known as **cancellation**. Small relative errors in the computation of $norm(x)$ will cause huge relative errors on $u_1$.

> **Key Insight:** To improve stability, a small modification is made: we choose $s = -\vert{}\vert{}x\vert{}\vert{}$ whenever $x_1 \ge 0$, and $s = \vert{}\vert{}x\vert{}\vert{}$ whenever $x_1 < 0$. In this way, the operation $x_1 - s$ always sums two numbers with the same sign, eliminating cancellation errors.

**Stable Solution:**
```
function [u, s] = householder_vector(x)
    s = norm(x);
    if x(1) >= 0, s = -s; end
    v = x;
    v(1) = v(1) - s;
    u = v / norm(v);
```

### The QR Factorization Theorem

> **Theorem:** For every matrix $A\in\mathbb{R}^{m\times n}$, there exists an orthogonal matrix $Q\in\mathbb{R}^{m\times m}$ and an upper triangular matrix $R$ (i.e., $i>j \Rightarrow R_{ij}=0$) such that $A = QR$. The most interesting case for this factorization is when $m \ge n$ (square or tall-thin matrices).

**Algorithm Steps via Householder Matrices:** We work column by column, using orthogonal matrices to transform $A$ into the upper triangular matrix $R$.

1. **Step 1:** Calculate the Householder reflector $H_1$ for the first column. Left-multiplying $H_1 A$ yields a new matrix $A_1$ with zeros below the first element of column 1.
    
2. **Step 2:** To introduce zeros in the second column without spoiling the ones already computed, we left-multiply by a block matrix $Q_2 = \begin{bmatrix} 1 & 0 \\ 0 & H_2 \end{bmatrix}$. This leaves the first row unchanged and multiplies the sub-matrix below by $H_2 \in\mathbb{R}^{(m-1)\times(m-1)}$.
    
3. **Completion:** After the $n$-th step ($n$ being the number of columns), we have a sequence of orthogonal matrices such that $Q_{n}\dots Q_{3}Q_{2}Q_{1}A = R$ is upper triangular. This implies the full factorization is $A = (Q_{1}^{T}Q_{2}^{T}\dots Q_{n}^{T})R$.
    

##### Optimizations for the Algorithm
As theoretically written, this algorithm would have a quartic cost ($\mathcal{O}(m^4)$) for a square matrix.
- **Huge Optimization:** Do not form the matrix $H$ explicitly. Instead, use the trick $HA_k = A_k - 2u(u^{T}A_k)$. This optimization brings down the cost from quartic to cubic.
- **Minor Optimization:** Write the $s$ values and zeros manually directly into the submatrix components of $A$.

### Rectangular "Thin" QR
If $m \gg n$ (like in SVD), computing or storing the full matrix $Q$ is highly expensive. **Thin QR:** We can restrict the calculation to a smaller scale $Q_0 \in \mathbb{R}^{m \times n}$ and $R_0 \in \mathbb{R}^{n \times n}$:

$$A = \begin{bmatrix} Q_0 & Q_c \end{bmatrix} \begin{bmatrix} R_0 \\ 0 \end{bmatrix} = Q_0 R_0$$

There are alternatives for handling $Q_0$ without forming the big matrix $Q$: You can just return the implicit form of the sequence of $u_j$ vectors. The implicit sequence $Q = Q_1 Q_2 \dots Q_n$ is not a dense array full of numbers, but you can still use it to perform operations such as matrix products at the same cost, or even cheaper.

### Computational Cost
The overall computational cost of thin QR factorization via Householder reflectors (assuming $m \ge n$) is:
$$2mn^2 - \frac{2}{3}n^3 + \mathcal{O}(mn) \text{ flops}$$
The behavior of this formula in two common regimes is essential:
1. **Square matrices ($m = n$):** The cost is approximately $\frac{4}{3}n^3$.
2. **Tall-thin matrices ($m \gg n$):** The cost scales like $2mn^2$.

## Least Squares Problems and QR Factorization

We see a different algorithm to solve least-squares problems using the QR factorization of A. To begin the derivation, start from:
$$A=QR , Q=[\begin{matrix}Q_{0}&Q_{c}\end{matrix}], R=[\begin{matrix}R_{0}\\ 0\end{matrix}]$$
Since orthogonal matrices preserve the 2-norm, we can rewrite the objective function:

$$\vert{}\vert{}Ax-y\vert{}\vert{}=\vert{}\vert{}Q^{T}(Ax-y)\vert{}\vert{}=\vert{}\vert{}Q^{T}QRx-Q^{T}y\vert{}\vert{}$$
$$=\vert{}\vert{}Rx-Q^{T}y\vert{}\vert{}=\vert{}\vert{}[\begin{matrix}R_{0}\\ 0\end{matrix}]x-[\begin{matrix}Q_{0}^{T}\\ Q_{c}^{T}\end{matrix}]y\vert{}\vert{}$$
$$=\vert{}\vert{}[\begin{matrix}R_{0}x-Q_{0}^{T}y\\ Q_{c}^{T}y\end{matrix}]\vert{}\vert{}$$

### Solving Least Squares with QR
After applying the orthogonal transformation, the problem becomes finding the minimum of:
$$\vert{}\vert{}Ax-y\vert{}\vert{}=\vert{}\vert{}[\begin{matrix}R_{0}x-Q_{0}^{T}y\\ Q_{c}^{T}y\end{matrix}]\vert{}\vert{}$$

How can we minimize the norm of this vector? The vector can be analyzed in two separate blocks:
- **Bottom block:** This has the same value, regardless of x. The squares of those entries will always be in the sum, representing the unavoidable residual error.
- **Top block:** We can choose x to make its entries smaller. Can we get $R_{0}x-Q_{0}^{T}y=0$ ? Yes, if $R_{0}$ is invertible.

##### When is $R_0$ invertible?
This is related to a result we have seen earlier. If $A=QR$, with Q orthogonal, then:
$$
A^{T}A = (QR)^{T}(QR) = R^{T}\underline{Q^{T}Q}R = R^{T}R = \begin{bmatrix} R_{0}^{T} & 0 \end{bmatrix} \begin{bmatrix} R_{0} \\ 0 \end{bmatrix} = R_{0}^{T}R_{0}
$$
Based on this derivation:
- If A has full column rank, ATA is posdef (positive definite).
- Because it is positive definite, it is invertible.
- Therefore, $R_{0}$ is invertible.

_(Note for your future self revising: $R_{0}^{T}R_{0}$ is the Cholesky factorization of $A^{T}A$, which we shall see later in the course.)_

### Algorithm
We have proved the following fundamental result.

> **Lemma:** If $A=QR=[\begin{matrix}Q_{0}&Q_{c}\end{matrix}][\begin{matrix}R_{0}\\ 0\end{matrix}]$ (and has full column rank), then the solution of $min\vert{}\vert{}Ax-y\vert{}\vert{}$ is given by $x=R_{0}^{-1}(Q_{0}^{T})y$.

The thin QR factorization $A=Q_{0}R_{0}$ contains all we need here to compute the solution.

> **Corollary:** This gives us the formula for the pseudoinverse: $A^{+}=R_{0}^{-1}Q_{0}^{T}$

##### Computational Cost
The computational cost to solve the problem is broken down into three steps:

1. **Thin QR:** $O(mn^{2})$
2. **Multiplication** $c=(Q_{0}^{T})b$: $O(mn)$.
3. **Triangular system solution** $R_{0}x=c:$ $O(n^{2})$ with back-substitution.

The dominant part of the overall cost is the thin QR computation.

### The Geometric Picture
Geometrically, the vector y is split into two orthogonal components: Ax and $y-Ax$.
- $Ax=Q_{0}R_{1}R_{0}^{-1}Q_{0}^{T}y=Q_{0}Q_{0}^{T}y$ gives the projection of y onto Im A. It has length $\vert{}\vert{}Q_{0}^{T}y\vert{}\vert{}$.
- The residual $\vert{}\vert{}Ax-y\vert{}\vert{}$ (i.e., the optimum value) is $\vert{}\vert{}Q_{c}^{T}y\vert{}\vert{}$.

These vectors visualize the Pythagorean theorem in the vector space: the vectors y, Ax and $Ax-y$ form a right triangle with side lengths $\vert{}\vert{}y\vert{}\vert{},\vert{}\vert{}Q_{0}^{T}y\vert{}\vert{},\vert{}\vert{}Q_{c}^{T}y\vert{}\vert{}$.