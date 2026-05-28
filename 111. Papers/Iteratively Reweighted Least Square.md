---
Data: 2026-05-02T21:32:00
Tags:
  - note
  - youngling
  - paper
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Iteratively Reweighted Least Square

The idea for these method is an alternative solution to resolve the [[Least Squares]] problem with an iteratively approach with any [[Vector Norms]].

## The Problem
We have to found the optimal approximation solution of a set of simultaneous linear equarions
$$
\begin{bmatrix}
a_{11}&a_{12}&\dots&a_{1N}\\
a_{21}&\dots\\
\vdots\\
a_{M1}& a_{M2} &\dots & a_{MN}
\end{bmatrix}
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_M
\end{bmatrix}
=
\begin{bmatrix}
y_1\\y_2\\\vdots\\y_M
\end{bmatrix}
$$
or, in matrix notation
$$
Ax=y
$$
and we want to find the $N$ by $1$ vector $x$. To resolve this [[Introduction to Linear Algebra#Linear systems|linear system]] if $y$ not lie in the range space of $A$  there is not exact solution, to fix these we transform in an approximation problem in this form:
$$
e = Ax-y
$$
and we define this that is the [[Least Squares]]:
$$
\arg\min ||e||^2_2 = e^Te
$$
that means **We want the value of x that minimize the suqare distance of e**.
###### Solution with $A$ non-singular matrix (Full Rank)
To do this we can use an Analytical solution only when $A$ is non-singular (its rows or columns are linearly independent).

- **If $A$ has $M = N$ (Square system):** In this case, there is a unique, exact solution. We do NOT need to minimize the Least Mean Square error because the error is exactly zero. Starting from the normal equations derived from minimizing $||e||_2^2$:
$$||e||^2_2 = e^Te = (Ax - y)^T (Ax-y)= x^TA^TAx - 2x^TA^Ty + y^Ty$$
    Set the derivative to zero to find the minimum:
$$A^TAx = A^Ty$$
    Since $A$ is square and full rank, it is invertible. We can multiply both sides by $(A^TA)^{-1}$:
$$x = (A^TA)^{-1}A^Ty = A^{-1}(A^T)^{-1}A^Ty \Rightarrow x = A^{-1}y$$
    
- **If $A$ has $M > N$ (Overdetermined system):** There are more equations than unknowns, so no exact solution exists. We must find the approximate solution that minimizes the Least Squared equation error ($L_2$ norm). Solving the normal equations $A^TAx = A^Ty$ for $x$:
$$x = [A^TA]^{-1}A^Ty$$
- **If $A$ has $M < N$ (Underdetermined system):** There are fewer equations than unknowns, leading to infinite exact solutions. We use the Least Norm approximation to find the specific solution where the variables $x$ are as close to zero as possible (minimum $L_2$ norm of the solution).
$$x = A^T[AA^T]^{-1}y$$
###### Solution with $A$ singular matrix (Not Full Rank)
When $A$ is singular, the matrices $[A^TA]$ or $[AA^T]$ have a [[Introduction to Linear Algebra#The Determinant|determinant]] of zero and cannot be inverted. The analytical formulas above lose their mathematical meaning and will fail (division by zero).
- **The Approach:** We abandon the standard analytical formulas and use a generalized solution.
- **[[Least Squares#Moore-Penrose Pseudoinverse|Moore-Penrose Pseudoinverse]]:** We use the pseudoinverse of $A$ (often denoted as $A^+$ or calculated via `pinv(A)` in software). This robustly finds the best $L_2$ approximation without requiring the matrix to be full rank.$$x = A^+y$$
This form is correct but we assume to use $L_2$ norm this because:
- In some cases that can be done by analytic formules
- $L_2$ norm has a energy interpretation

However, both the $L_1$ and $L_{\infty}$ norms have well know applications that are important and use the more general $L_p$ error is remarkably flexible. This is the goal of this paper. Many of these norms are not directly differentialbe, so it use a different approach.

If the goal is to minimize a generic $L_p$ norm (e.g., $L_1$ for sparse solutions or $L_\infty$ to control maximum error) instead of the standard $L_2$ norm, direct analytical formulas do not exist.

## The $L_p$ Norm Approximation
First we have to define a variant of the classic Least Square analytic solution that introduce the more general concept of weight matrix that can emphasise or de-emphasise certain components:
$$
\arg\min ||We||^2_2 = e^TW^T We
$$
where $W$ is a diagonal matrix with weights $w_i$ along its diagonal
- if **A** has $M>N$
$$
x = [A^TW^TWA]^{-1} A^TW^T Wy
$$
- id **A** has $M<N$ 
$$
x = [W^TW]^{-1} A^T \big[A [W^TW]^{-1} A^T \big]^{-1}y
$$
**Note:** for the case $M=N$ we simply resolve the system $x=A^{-1}y$

#### Core idea
First we define the $L_p$ equation error norm as:
$$
||e||_p = \bigg(\sum_n |e(n)|^p\bigg)^{1/p}
$$
and finding $x$ to minimizing the p-norm equation.

Now we now very well how to resolve the $L_2$, and also, as we seen above, the $L_2$ with $W$ parameter. So the idea is to converge on that problem easy to resolve in a iteratively way.

First, instad to resolve $||e||_p$ we will resolve $||e||_p^p$ that is the same but it remove the square componet making the calculus more siply

1. Split the $|e(n)|^p$
$$
||e||^p_p = \sum_n |e(n)|^{p-2} |e(n)|^2
$$
2. We define the following things:
$$
w(n)^2 = |e_n|^{p-2} \Leftrightarrow w(n) = |e_n|^{\frac{p-2}{2}}
$$
3. Now we can define the split in 1 as the equation above:
$$
||e||^p_p = \sum_n |e(n)|^{p-2} |e(n)|^2 = \sum_n |w(n)|^2|e(n)|^2 \Longrightarrow e^TW^t Ww
$$
from this we do the derivate and go along the M and N relations we calculate the result for this iteration and than and than we move to the next until we achice a final $||e||^2_2$ so No decoposizions

There is just one problem, $W$ depends to $x$ and $x$ depends to $W$ to we need a **Start condition**. We calculate the x matrix as:
$$
x = pinv(A) \cdot y
$$
where $pinv$ is the [[Least Squares#Moore-Penrose Pseudoinverse|Moore-Penrose Pseudoinverse]] to remove all the problem for singular matrix, because from this moment onwards all calculus are with no-singular.

#### The Overdetermined System M>N
In this case we do not minimize $e$ because it doesn't make sense,  we have **infinite possible solutions** so, the error must be always 0. We need to found the minimum solution.

```
% Basic IRLS Algorithm for M > N
x = pinv(A)*y;                   % 1. Initial L_2 solution using Moore-Penrose
for k = 1:KK                     % 2. Start the iterative loop
    e = A*x - y;                 % 3. Calculate current Error vector
    w = abs(e).^((p-2)/2);       % 4. Calculate Error weights based on p-norm
    W = diag(w/sum(w));          % 5. Create normalized diagonal weight matrix
    WA = W*A;                    % 6. Apply weights to matrix A
    x = (WA'*WA)\(WA'*W)*y;      % 7. Calculate new weighted L_2 solution
end
```

**Step-by-step Explanation:**

1. **Initial Solution:** The algorithm starts by finding a standard $L_2$ approximation using the pseudoinverse (`pinv(A)*y`). This provides the baseline $x$ to calculate the first error.
    
2. **Iterative Loop:** The core process repeats for `KK` iterations (until the result stops changing significantly).
    
3. **Error Calculation:** At each step, it calculates how far the current solution $x$ is from the target $y$ by computing $e = Ax - y$.
    
4. **Weight Update:** This is the core IRLS trick. It generates new weights $w(n)$ using the formula derived above: $|e|^{\frac{p-2}{2}}$. This means the weight of each equation dynamically changes based on its current error and the chosen $p$ value.
    
5. **Matrix Construction:** The weights are normalized and placed on the diagonal of matrix $W$.
    
6. **Application:** The weights are applied to the original matrix $A$.
    
7. **Weighted $L_2$ Solve:** It calculates the new $x$ using the weighted least squares formula for overdetermined systems. In Matlab syntax, `(WA'*WA)\(WA'*W)*y` is the computational equivalent of $x = [A^TW^TWA]^{-1} A^TW^TWy$, obtaining the new solution to feed into the next loop.
    
### The Underdetermined System ($M < N$)
In this scenario, we have more unknowns than equations (infinite exact solutions exist). Instead of minimizing the equation error (which is exactly zero), we want to find the specific $x$ that satisfies $Ax = y$ while minimizing the $p$-norm of the solution vector $x$ itself ($||x||_p$).

Here is the basic Matlab algorithm implementation for IRLS in this case:
```
% Basic IRLS Algorithm for M < N
x = pinv(A)*y;                            % 1. Initial L_2 solution (Least Norm)
for k = 1:KK                              % 2. Start the iterative loop
    W = diag(abs(x).^((2-p)/2)+0.00001);  % 3. Calculate norm weights for x
    AW = A*W;                             % 4. Apply new weights
    x = W*AW'*((AW*AW')\y);               % 5. Calculate new Weighted L_2 solution
end
```

**Step-by-step Explanation:**

1. **Initial Solution:** Again, it starts with the Moore-Penrose pseudoinverse (`pinv(A)*y`). For $M < N$, this automatically provides the "Least Norm" $L_2$ solution (the solution with the smallest Euclidean length).
    
2. **Iterative Loop:** The loop begins.
    
3. **Weight Update (Crucial Difference):** Instead of calculating weights based on the equation error $e$, the weights are now calculated directly from the solution vector $x$. The formula is $w = |x|^{\frac{2-p}{2}}$. A small value (`+0.00001`) is added to prevent infinite weights (division by zero) in case any element of $x$ becomes exactly zero. These weights are then placed on the diagonal of matrix $W$.
    
4. **Application:** The matrix $A$ is multiplied by the weight matrix $W$.
    
5. **Weighted Solve:** It computes the new solution using the weighted formula for underdetermined systems:
    
    $$x = [W^TW]^{-1} A^T \big[A [W^TW]^{-1} A^T \big]^{-1}y$$
    
    In Matlab, this is efficiently computed as `W*AW'*((AW*AW')\y)`. This new $x$ becomes the input for the next iteration.
# References
- [[Iterative_Reweighted_Least_Squares.pdf]]