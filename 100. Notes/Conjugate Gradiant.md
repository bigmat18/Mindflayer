---
Data: 2026-04-05T12:11:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area:
---
# Conjugate Gradiant
Solving systems of equations Storing an $m\times m$ matrix with $m = 100 000$ requires ≈ 80GB. And even if we managed to do it, applying an algorithm like Gaussian elimination, with complexity $O(m^3)$, is prohibitive.

Luckily, many real-world matrices are **sparse**: for instance 3, 10 nonzeros per row. This includes matrices from graph/networks, KKT systems,
discretization of differential equations. 

- Adjacency matrices from **networks** and **graphs**. For instance, in some applications centrality indices are computed by solving $(I-\alpha A)x = ones (n, 1)$

```
% ’’Friendship matrix’’ on a group of 34 people
M = load(’karate.mat’).Problem.A;
% Road network of Luxembourg
M = load(’luxembourg_osm.mat’).Problem.A;
```

- In both **engineering** and **video game programming**, one often models complex objects as “networks of points joined by forces”, and then solves problems on them.
```
% From a structural stability problem (Boeing)
M = load(’msc00726.mat’).Problem.A;
```

- KKT matrices in optimization, $\begin{bmatrix}D_1&A\\A^T&0\end{bmatrix}$, often with D1 diagonal and tall-thin A (possibly already sparse).

### Storing and using a sparse matrix
Basic format to store sparse matrices: as a list of non-zero $(i, j, A_{ij})$.
```
 sprandn(10,10,0.3)
```
    
**Detail:** if the indices $j$ are listed increasingly, they can be compressed further (CSC/CSR - compressed sparse column/row).
    
Python pseudocode to compute the product $w=A*v$:
```python
def compute_product(A, v):
    w = zeros(size(A, 1))
    for (i, j, Aij) in A:
        w[i] += Aij * v[j]
    return w
```

### Optimization to solve linear systems
Given an $n\times n$ matrix $Q>0$ and a vector $v=-q\in\mathbb{R}^{n}$, we wish to minimize:
$$\min f(x)=\frac{1}{2}x^{T}Qx-v^{T}x+\text{const.}$$

This is equivalent to solving $g=Qx-v=0$, i.e., the **linear system** $Qx=v$. The algorithm we use computes at each step the best possible approximation $x_{k}$ to the solution $x_{*}$. It is particularly suited to large problems with **sparse** matrices.

## Conjugate gradient
Let us start from a simple quadratic problem with $Q=I$:
$$\min_{y\in\mathbb{R}^{m}}\frac{1}{2}||y-w||^{2}+\text{const}=\min\frac{1}{2}y^{T}y-w^{T}y+\text{const}$$
$$
= \min \frac{1}{2} (y_1^2 + y_2^2 + \dots + y_m^2) - (w_1y_1 + w_2y_2 + \dots + w_m y_m)
$$
This problem is separable: starting from $y_{0}=0$, we optimize on **each coordinate** separately and generate the sequence of vectors:
$$
y_1 = \begin{bmatrix}w_1\\0\\0\\\vdots\\0\end{bmatrix}, 
y_2 = \begin{bmatrix}w_1\\w_2\\0\\\vdots\\0\end{bmatrix}, 
y_3 = \begin{bmatrix}w_1\\w_2\\w_3\\\vdots\\0\end{bmatrix}, \dots
$$

At each step, we add a multiple of a new search direction $e_{1},e_{2},e_{3},\dots$ which are all orthogonal to each other. Convergence is guaranteed after $m$ iterations. 

### Subspace Optimality
At each step, we solve a 1D problem and choose $y_k$ to solve:
$$
y_k = \arg\min f(y) \:\:\: \text{ over } \:\:\: \begin{bmatrix}w_1 \\ \vdots \\ w_{k-1} \\ * \\ 0\\ \vdots \\ 0 \end{bmatrix}
= \{y_{k-1} + \alpha e_k : \alpha \in \mathbb{R}\}
$$
line search), but we also get for free a stronger property:
$$
y_k = \arg\min f(y) \:\:\: \text{ over } \:\:\: \begin{bmatrix}* \\ \vdots \\ * \\ * \\ 0\\ \vdots \\ 0 \end{bmatrix}
= \text{span} (e_1, \dots, e_k)
$$
### Orthogonal directions and The Algorithm
We can proceed similarly with any set of **orthogonal search directions** $U=[u_{1},u_{2},\dots,u_{m}]$ (i.e., $u_{i}^{T}u_{j}=0$ when $i\ne j$) instead of the canonical basis $e_1, e_2, \dots, e_m$
$$
w = U \begin{bmatrix}c_1 \\ c_2 \\ \vdots \\ c_m\end{bmatrix}, \:\:\:\: ||w|| = ||c||
$$
and find
$$
y_k = \arg\min f(y) \:\:\: \text{ over } \:\:\: \begin{bmatrix}w_1 \\ \vdots \\ w_{k-1} \\ * \\ 0\\ \vdots \\ 0 \end{bmatrix}
= \{y_{k-1} + \alpha e_k : \alpha \in \mathbb{R}\}
$$
$$
= \arg\min f(y) \:\:\: \text{ over } \:\:\: \begin{bmatrix}* \\ \vdots \\ * \\ * \\ 0\\ \vdots \\ 0 \end{bmatrix}
= \text{span} (e_1, \dots, e_k)
$$
Given orthogonal search dirs $u_1, \dots, u_m$ (i.e. $u_i^T u_j = 0$ when $i\neq j$)
```
y0 <- 0
for k=1,2,3,...,m do
	y_k <- argmin||y-w||^2 + const over {y_{k-1} + alpha u_k};
end
```

### Change of variable
This problem is equivalent to any quadratic problem via a change of basis: given $R\in\mathbb{R}^{m\times m}$ invertible, $y=Rx$.

$$
\min \frac{1}{2} y^Ty - w^T y + \text{ const } = \min \frac{1}{2} x^T R^T R x - w^T R x + \text{const}
$$
where $R^T R = Q$ and $w^T R = v^T$

We can solve the (difficult) problem on the x-space by looking at
the (easier) one on the y-space, with coordinate descent.

**Important detail:** in the old problem, $w$ is both the **linear term** appearing in the objective function and the solution $y_∗ = w$; in the new problem, $v = R^T w$, but $x_∗ = Q^{−1}v = R^{−1}w$: indeed we can rewrite the objective function as
$$\min_{y}\frac{1}{2}||y-w||^{2}+C=\min_x\frac{1}{2}(x-x_{*})^{T}R^{T}R(x-x_{*})$$

> **Definition**: The Q-norm of a vector $z$ is $||z||_{Q}=(z^{T}Qz)^{1/2}$. Since $Q>0$, $||z||_{Q}\ge0$, with equality iff $z=0$. Search directions $Rd_{k}=u_{k}$ become orthogonal in the y-space, meaning $d_{i}^{T}R^{T}Rd_{j}=0$, which equals $d_{i}^{T}Qd_{j}=0$. 


##### Q-orthogonality
Search directions: $Rd_k = u_k$ . These are orthogonal in the y-space
($u^T_i u_j = 0$ when  $i \neq j$), but in the y-space the relation becomes
$$
d_j^T R^T R d = 0
$$
> **Definition**: Vectors $d_{i}, d_{j}$ are called Q-orthogonal if $d_{i}^{T}Qd_{j}=0$.


Here are the complete notes formatted cleanly in Markdown, with your initial sections refined to use appropriate LaTeX formatting and code blocks, followed by the missing sections from the document.

### The Algorithms (x-space vs y-space)
###### In the y-space
Given orthogonal search dirs $u_{1},...,u_{m}$ (i.e., $u_{i}^{T}u_{j}=0$ when $i\ne j$):

```
y_0 <- 0;
for k = 1, 2, 3, ..., m do
    y_k <- arg min ||y-w||^2 + const over {y_{k-1} + alpha u_k}; 
    // univariate quadratic problem in alpha
end
```

###### In the x-space
Given Q-orthogonal search dirs $d_{1},...,d_{m}$ (i.e., $d_{i}^{T}Qd_{j}=0$ when $i\ne j$):

```
x_0 <- 0;
for k = 1, 2, 3, ..., m do
    x_k <- arg min x^T Q x + v^T x + const over {x_{k-1} + alpha d_k}; 
    // univariate quadratic problem in alpha
end
```

Some details about these algorithms:

- We do not need to know $R$, nor $x_{*}$, nor const: it is enough to have $Q$ and $v$!
- **Subspace optimality:** $x_{k}=\min~f(x)$ for $x\in \text{span}(d_{1},...,d_{k})$. Convergence guaranteed in $m$ steps, but we hope to do better!
- **Important missing part:** How to choose the $d_j$'s? Optimization suggests it should be loosely in the direction of the residual $r_{j}=-g_{j}=v-Qx_{j}$. But residuals are not Q-orthogonal.
- We shall see that a special property holds: if we set $d_{j}=r_{j}+\beta_{j}d_{j-1}$, it is sufficient to choose $\beta_{j}$ to impose $d_{j-1}^{T}Qd_{j}=0$. Q-orthogonality with all previous search directions holds automatically.
    

### Conjugate Gradient Implementation
The algorithm requires three ingredients: the current iterate $x_{j}$, the residual $r_{j}=v-Qx_{j}=-g_{j}$, and the search direction $d_j$.

**CG iteration block:**
```
x_0 = 0, r_0 = d_0 = v;
for j = 1:n do
    alpha_j = (r_{j-1}^T r_{j-1}) / (d_{j-1}^T Q d_{j-1}); // exact line search
    x_j = x_{j-1} + alpha_j d_{j-1};
    r_j = r_{j-1} - alpha_j Q d_{j-1}; // residual update (check!)
    beta_j = (r_j^T r_j) / (r_{j-1}^T r_{j-1});
    d_j = r_j + beta_j d_{j-1}; // Q-orthogonal (we'll see why)
end
```

- The formula for the exact line search $\alpha_{j}$ is not obvious, but we will have the tools to prove it later.
- **Storage:** 3 vectors: $x_{j}$, $r_{j}$, $d_{j}$. There is no need to keep previous iterates.
    
###### Black-box Algorithms
- **Cost:** $n\times$ (1 mat-vec product for $Qd_{j-1}+\mathcal{O}(m))$.
- **Dominant part:** computing $n$ products $d_{j}\mapsto Qd_{j}$.
- Note that we only need a function ("oracle" in CS terms) `compute_product(d) = Q*d`: this makes it a so-called black box algorithm.
- CG is fast whenever `compute_product` is fast: a sparse $Q$ yields $\mathcal{O}(\text{nnz}(Q))$, but not only.
    

## Krylov Spaces
A new linear algebra concept that will help us analyze CG.

> **Definition:** Given $Q\in\mathbb{R}^{m\times m}$ (not nec. symmetric), $v\in\mathbb{R}^{m}$, and $n\le m$, the Krylov space $K_{n}(Q,v)$ is the linear subspace:$$K_{n}(Q,v)=\text{span}(v,Qv,Q^{2}v,...,Q^{n-1}v);$$
    
That is, the set of vectors that we can write as:$$w=(c_{0}I+c_{1}Q+c_{2}Q^{2}+\cdot\cdot\cdot+c_{n-1}Q^{n-1})v;$$which equates to any polynomial of degree $d<n$ in $Q$, multiplied by $v$.
    
### Polynomials and degrees
**Polynomials and degrees** Assume $v,Qv,Q^{2}v,...,Q^{n-1}v$ are linearly independent; then the coordinates 
$$
w = v c_0 + Qvc_1 + Q^2 vc_2 + \dots + Q^{n-1} vc_{n-1}
$$
of any vector $w\in K_{n}(Q,v)$ are unique. For each $w$ the degree $d$ of the polynomial such that $w=p(Q)v$ is well-defined.

- If $w$ has degree $d$, then $w\in K_{d+1}(Q,v)\backslash K_{d}(Q,v)$.
- If $w$ has degree $d$, then $Qw$ has degree $d+1$.
    
### Krylov spaces characterization
$K_{n}(Q,v)$ is the set of vectors that I can obtain, starting from $S=\{v\}$, with these operations:
1. **Multiply by Q:** add to the set $Qw$, where $w$ is any element of $S$.
2. **Linear combinations:** add to the set $w_{1}\alpha_{1}+\cdot\cdot\cdot+w_{k}\alpha_{k}$, where the $w_{i}$ belong to $S$.
    

_Condition: the first operation is performed fewer than $n$ times_. This matches well our "oracle" idea: the allowed operations are linear combinations and invoking the oracle; $K_{n}(Q,v)$ is the set of vectors that I can obtain by calling the oracle fewer than $n$ times.

### Krylov Spaces and Optimization
The iterates of gradient descent lie in Krylov spaces. Suppose we are looking for:
$$\min~f(x)=\frac{1}{2}x^{T}Qx-v^{T}x+\text{const}, \quad x_{0}=0$$
At each step we take a gradient $g_{k}:=Qx_{k}-v$ and use it to compute $x_{k+1}$:
- $x_{0}=0$
- $x_{1}=x_{0}-(Qx_{0}-v)\alpha_{1}=v\alpha_{1}$ (0 products with $Q$ required)
- $x_{2}=x_{1}-(Qx_{1}-v)\alpha_{2}=v\alpha_{1}-(Qv\alpha_{1}-v)\alpha_{2}\in \text{span}(v,Qv)$ (1 product with $Q$ required)
- $x_{3}=x_{2}-(Qx_{2}-v)\alpha_{3}\in \text{span}(v,Qv,Q^{2}v)$ (2 products with $Q$ required)

We have:
- $g_0, x_{1}\in K_{1}(Q,v)$
- $g_1, x_{2}\in K_{2}(Q,v)\backslash K_{1}(Q,v)$
- $g_2, x_{3}\in K_{3}(Q,v)\backslash K_{2}(Q,v)$

### Search Space = Krylov Space

> **Theorem:** Assume that $v,Qv,...,Q^{n-1}v$ are linearly independent. Then, after each step $n$ of CG (starting from $x_{0}=0)$,
> - $x_{1},x_{2},...,x_{n}$
> - $r_{0},r_{1},...,r_{n-1}$
> - $d_{0},d_{1},...,d_{n-1}$ 
> 
> are bases of $K_{n}(Q,v)$.
    

**Proof:**
1. Using the formulas that define the method, show inductively that $x_{j},r_{j-1},d_{j-1}$ have degree $j-1$.
2. Observe that if we have a polynomial $p_{0}(t)$ of degree 0, one $p_{1}(t)$ of degree 1, ..., one $p_{n-1}(t)$ of degree $n-1$, then we can write any polynomial of degree $\le n-1$ as a linear combination of them.
    

### Orthogonality in CG

> **Theorem:** At each step $r_{i}^{T}r_{j}=d_{i}^{T}Qd_{j}=0$ for all $i<j$. The $r_{i}$ are orthogonal (not ortho-normal), and the $d_i$ are Q-orthogonal.

**Proof (sketch):** Assume it holds for $j-1$ (induction!). We show only that $r_{i}^{T}r_{j}=0$ for all $i<j$; the other part is similar. From $r_{j}=r_{j-1}-\alpha_j Qd_{j-1}$ it follows that:
$$r_{i}^{T}r_{j}=r_{i}^{T}r_{j-1}-\alpha_{j}r_{i}^{T}Qd_{j-1}$$

- For $i<j-1$, $r_{i}^{T}r_{j-1}$ is zero by induction, since $r_{i}\in \text{span}(d_{0},d_{1},...,d_{i})$ is Q-orthogonal to $d_{j-1}$.
- For $i=j-1$, the RHS is zero if we can prove that $\alpha_{j}=\frac{r_{j-1}^{T}r_{j-1}}{r_{j-1}^{T}Qd_{j-1}}$. This is almost the formula for $\alpha_{j}$, but the denominator is not $d_{j-1}^{T}Qd_{j-1}$. However $d_{j-1}=r_{j-1}+\beta_{j-1}d_{j-2}$ so the two denominators differ by $\beta_{j-1}d_{j-2}^{T}Qd_{j-1}=0$ (by induction).
    

_(For Completeness)_: It remains to prove the second half of the induction step, i.e., $0=d_{i}^{T}Qd_{j}=d_{i}^{T}Q(r_{j}+\beta_{j}d_{j-1})$.

- For $i<j-1$, this follows by induction and the fact that $Qd_{i}\in K_{j-1}(Q,v)$ is orthogonal to $r_{j}$.
    
- For $i=j-1$, this holds if we can prove that:
    
    $$\beta_{j}=-\frac{d_{j-1}^{T}Qr_{j}}{d_{j-1}^{T}Qd_{j-1}}=\frac{r_{j}^{T}(-\alpha Qd_{j-1})}{d_{j-1}^{T}(\alpha Qd_{j-1})}=\frac{r_{j}^{T}(r_{j}-r_{j-1})}{d_{j-1}^{T}(r_{j-1}-r_{j})}$$
    
    This quantity is equal to the formula for $\beta_{j}$ because $d_{j-1},r_{j-1}\in K_{j}(Q,v)$ are orthogonal to $r_{j}$, and $d_{j-1}^{T}r_{j-1}=(r_{j-1}+\beta_{j-1}d_{j-2})^{T}r_{j-1}=r_{j-1}^{T}r_{j-1}$.
    

### Lucky Breakdown
**Breakdown = solution:** Suppose that for a certain $n$ the vectors $v, Qv, \dots, Q^n v$ are linearly dependent, i.e., $Q^{n}v$ can be written as a linear combination of the previous ones, meaning 
$$
K_{n}(Q,v)=K_{n-1}(Q,v)
$$
In particular, since $r_{n}\in K_{n}(Q,v)$ we have:
$$r_{n}=c_{0}r_{0}+c_{1}r_{1}+\cdot\cdot\cdot+c_{n-1}r_{n-1}$$

But we can still use the steps of our proof to show that $r_{n}^{T}r_{j}=0$ for $j<n$. Then, we must have $||r_{n}||^{2}=r_{n}^{T}r_{n}=0$, by orthogonality.

### Convergence of CG
- **Geometric idea:** the level curves are ellipsoids in the x-space but circles in the y-space.
- Convergence is guaranteed in at most $m$ iterations, $x_{m}=x_{*}$, but it can be much faster. For instance, when $Q=I$ we converge in 1 step.
- **Optimality:** $\Rightarrow||x_{k}-x_{*}||_{Q}$ and $f(x_{k})$ decrease monotonically. However, $||x_{k}-x_{*}||$ or $||r_{k}||$ do not, in general.
- **Optimality:** $\Rightarrow||x_{k}-x_{*}||_{Q}$ and $f(x_{k})$ decrease faster than any other method that produces $x_{n}\in K(Q,v)$. E.g., gradient method, heavy ball variants.
    

### Convergence speed of CG
The convergence speed depends on the effectiveness of polynomial approximation of the eigenvalues of $Q$.

> **Theorem:**
> $$\frac{||x_{n}-x_{*}||_{Q}}{||x_{0}-x_{*}||_{Q}}\le \min_{r(t)}\max_{i=1,2,...,m}|r(\lambda_{i})|$$
> where $\lambda_{1},...,\lambda_{m}$ are the eigenvalues of $Q$, and the minimum is over all polynomials $r$ of degree $\le n$, normalized such that $r(0)=1$.

**Proof:**
$x_{n}\in K_{n}(Q,v)\iff x_{n}=p(Q)v$ for a polynomial $p$ of degree $<n$.
$$||x_{n}-x_{*}||_{Q}=\min_{x\in K_{n}(Q,v)}||x-x_{*}||_{Q}=\min_{p(t)}||x_{*}-p(Q)Qx_{*}||_{Q}$$
$$=\min_{r(t)=1-tp(t)}||r(Q)x_{*}||_{Q}$$
We can use the formulas from our slideset on orthogonality to give better expressions in terms of an eigenvalue decomposition $Q=UDU^{T}$, with $U$ orthogonal and $D$ diagonal containing the eigenvalues:

$$r(Q)=U \begin{bmatrix} r(\lambda_{1}) & & \\ & \ddots & \\ & & r(\lambda_{m}) \end{bmatrix} U^{T}$$

Moreover, if $x_{*}=Uc$ then $||x_{*}||_{Q}^{2}=\sum_{i}\lambda_{i}c_{i}^{2}$, and

$$||r(Q)x_{*}||_{Q}^{2} = \sum_{i}\lambda_{i}r(\lambda_{i})^{2}c_{i}^{2}$$

From these two expressions it follows that:

$$\frac{||x_{n}-x_{*}||_{Q}^{2}}{||x_{0}-x_{*}||_{Q}^{2}}=\frac{||r(Q)x_{*}||_{Q}^{2}}{||x_{*}||_{Q}^{2}}$$

$$=\frac{\lambda_{1}r(\lambda_{1})^{2}c_{1}^{2}+\cdot\cdot\cdot+\lambda_{m}r(\lambda_{m})^{2}c_{m}^{2}}{\lambda_{1}c_{1}^{2}+\cdot\cdot\cdot+\lambda_{m}c_{m}^{2}} \le \max_{\lambda_{1},...,\lambda_{m}}r(\lambda_{i})^{2}$$

### CG finds the best polynomial
CG converges as well as the best possible polynomial $r(t)$; and we don't even need to compute it explicitly!

![[Pasted image 20260406000003.png | 350]]

##### Repeated eigenvalues
If $Q\in\mathbb{R}^{m\times m}$ has only $n<m$ distinct eigenvalues, then we can find $r(t)$ such that $r(\lambda_{i})=0, r(0)=1$, by interpolation.
$$||x_{n}-x_{*}||_{Q}=0$$
$\Rightarrow$ CG finds the exact solution in $n$ steps!

![[Pasted image 20260406000040.png | 350]]

##### Clustered eigenvalues
**Similar case:** if the eigenvalues of $Q$ are clustered around $n$ values $\mu_{1},...,\mu_{n}$, then the interpolation polynomial $r$ on the $\mu_{i}$ is likely to have small $|r(\lambda_{i})|$ for all $i$$\Rightarrow$ small residual after $n$ steps.

![[Pasted image 20260406000119.png | 350]]

### Linear convergence
> **Theorem (linear convergence)**
> Let $\lambda_{max}$, $\lambda_{min}$ be the maximum/minimum eigenvalue of $Q$; then, CG converges with rate:
> $$\frac{||x_{n}-x_{*}||_{Q}}{||x_{0}-x_{*}||_{Q}}\le2\left(\frac{\sqrt{\lambda_{max}}-\sqrt{\lambda_{min}}}{\sqrt{\lambda_{max}}+\sqrt{\lambda_{min}}}\right)^{n}$$

_(Proof: find a Chebyshev-like polynomial such that $\max_{\lambda\in[\lambda_{min},\lambda_{max}]}|r(\lambda)|=$ RHS)_.

We can rewrite that constant in terms of $\kappa(Q)=\frac{\lambda_{max}}{\lambda_{min}}$, the **condition number** of $Q$ (this definition is valid only for $Q>0!$). This quantity measures how "imbalanced" the eigenvalues of $Q$ are.
$$\frac{\sqrt{\lambda_{max}}-\sqrt{\lambda_{min}}}{\sqrt{\lambda_{max}}+\sqrt{\lambda_{min}}}=\frac{\sqrt{\kappa(Q)}-1}{\sqrt{\kappa(Q)}+1}$$
This is a faster rate than that of the gradient method, which is $\frac{\kappa-1}{\kappa+1}$.

# References