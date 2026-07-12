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
First, we have to define a variant of the classic Least Squares analytic solution that introduces the more general concept of a weight matrix. This matrix allows us to emphasize or de-emphasize certain components:
$$\arg\min \Vert{}We\Vert{}^2_2 = e^T W^T W e$$
where $W$ is a diagonal matrix with weights $w_i$ along its diagonal.

### Case 1: Overdetermined System ($M > N$)
In this scenario, there are more equations than unknowns, so no exact solution exists ($e \neq 0$). We must find the vector $x$ that minimizes the weighted squared error.

**Objective Function:** Since the error is defined as $e = Ax - y$, we substitute this into our objective:
$$f(x) = (Ax - y)^T W^T W (Ax - y)$$
###### Step 1: Algebraic Expansion
Applying the transpose property $(Ax - y)^T = x^T A^T - y^T$ and expanding the terms:
$$f(x) = (x^T A^T - y^T) W^T W (Ax - y)$$
$$f(x) = x^T A^T W^T W A x - x^T A^T W^T W y - y^T W^T W A x + y^T W^T W y$$
Since $x^T A^T W^T W y$ is a scalar (a single number), it is equal to its transpose $y^T W^T W A x$. We can combine the middle terms:
$$f(x) = x^T A^T W^T W A x - 2x^T A^T W^T W y + y^T W^T W y$$
###### Step 2: Derivative and Optimization
To find the global minimum, we compute the gradient of $f(x)$ with respect to $x$ and set it to zero:
- The derivative of the quadratic form $x^T (A^T W^T W A) x$ is $2 A^T W^T W A x$.
- The derivative of the linear term $- 2x^T A^T W^T W y$ is $- 2 A^T W^T W y$.
- The term $y^T W^T W y$ is a constant with respect to $x$, so its derivative is $0$.
$$\nabla_x f(x) = 2 A^T W^T W A x - 2 A^T W^T W y = 0$$
###### Step 3: Solving the Linear System
Divide by 2 and isolate $x$:
$$A^T W^T W A x = A^T W^T W y$$
Multiplying both sides by the inverse of the matrix term attached to $x$ yields the final formula:
$$x = [A^T W^T W A]^{-1} A^T W^T W y$$

### Case 2: Underdetermined System ($M < N$)
In this scenario, there are fewer equations than unknowns, meaning there are infinite exact solutions where the error is exactly zero ($Ax - y = 0$). Minimizing the error no longer makes sense. Instead, we want to find the specific exact solution $x$ that has the smallest possible weighted norm (size).

###### Step 1: Method of Lagrange Multipliers
We construct the Lagrangian function $\mathcal{L}$ to incorporate the exact constraint into the objective function. We introduce the vector of multipliers $\lambda$ (adding a factor of $2$ purely for algebraic convenience during the derivation):
$$\mathcal{L}(x, \lambda) = x^T W^T W x + 2\lambda^T (y - Ax)$$
###### Step 2: Derivative with respect to $x$
Compute the gradient of the Lagrangian with respect to $x$ and set it to zero to find the minimum:
$$\nabla_x \mathcal{L} = 2 W^T W x - 2 A^T \lambda = 0$$
Divide by 2 and isolate $x$:
$$W^T W x = A^T \lambda$$
$$x = [W^T W]^{-1} A^T \lambda$$
###### Step 3: Solving for the multipliers $\lambda$
We now have an expression for $x$, but we need to find the value of $\lambda$. Substitute the expression for $x$ back into our original constraint $Ax = y$:
$$A \left( [W^T W]^{-1} A^T \lambda \right) = y$$
$$\left( A [W^T W]^{-1} A^T \right) \lambda = y$$
Since the bracketed term is an invertible $M \times M$ square matrix, we can isolate $\lambda$:
$$\lambda = \left[ A [W^T W]^{-1} A^T \right]^{-1} y$$
###### Step 4: Final Substitution
Finally, insert the value of $\lambda$ back into the formula for $x$ from Step 2:
$$x = [W^T W]^{-1} A^T \big[A [W^T W]^{-1} A^T \big]^{-1} y$$

**Note:** For the special case where $M = N$ (a square, full-rank system), no optimization process is needed. The matrix $A$ is directly invertible, and the solution is simply the exact intersection:
$$x = A^{-1}y$$
### Core idea
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

### Algorithm
##### The Overdetermined System M>N
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
    
##### The Underdetermined System ($M < N$)
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


## Optimization via Push-Through Identity
In standard regression settings, solving the Iteratively Reweighted Least Squares (IRLS) problem involves directly updating the primal system at each iteration. However, when transitioning to Machine Learning applications like a sparse [[Extreme Learning Machine]], where we combine a Least Squares error with an $\ell_1$ penalty (Lasso), we face a critical computational bottleneck in the underdetermined regime ($M < N$).

The primal system requires finding the parameter weights by solving the following linear system:

$$(X^T X + 2\lambda W_k^2) w_{k+1} = X^T y$$

To find $w_{k+1}$, we would theoretically need to invert the $N \times N$ matrix on the left:

$$w_{k+1} = (X^T X + 2\lambda W_k^2)^{-1} X^T y$$

When deployment involves a massive hidden layer ($N \gg M$), computing the inverse of this $N \times N$ matrix becomes computationally prohibitive, Scaling at $\mathcal{O}(N^3)$.

To circumvent this issue and shift the calculation into the smaller $M \times M$ sample space, we apply the **Push-Through Matrix Identity**. This identity states that for any matrices $X \in \mathbb{R}^{M \times N}$ and a non-singular diagonal matrix $D \in \mathbb{R}^{N \times N}$, the following algebraic equivalence holds exactly:

$$(X^T X + \lambda D)^{-1} X^T \equiv D^{-1} X^T (X D^{-1} X^T + \lambda I)^{-1}$$

By mapping our primal ELM system to the Push-Through Identity, we can substitute $D = 2W_k^2$ and match the regularization terms:
###### Step 1: Apply the identity to the primal inverse block
We rewrite the entire inverse and projection sequence $(X^T X + 2\lambda W_k^2)^{-1} X^T$ by pushing the matrix $X^T$ through:
$$w_{k+1} = \underbrace{(2W_k^2)^{-1} X^T \big[X (2W_k^2)^{-1} X^T + \lambda I\big]^{-1}}_{\text{Pushed-Through Identity Equivalent}} y$$
###### Step 2: Factor out the scalar constants
We can factor out the constant $2$ from the matrix inverses to align with our stabilization framework:
$$w_{k+1} = \frac{1}{2} W_k^{-2} X^T \big[\frac{1}{2} X W_k^{-2} X^T + \lambda I\big]^{-1} y$$
Multiplying the inside of the inverted bracket by $2$ and balancing it outside the bracket allows us to eliminate the fractions:
$$w_{k+1} = W_k^{-2} X^T \big[X W_k^{-2} X^T + 2\lambda I\big]^{-1} y$$
###### Step 3: Define the Matrix Substitution $P_k$
The expression still contains the awkward term $W_k^{-2}$, which represents the mathematical inverse of our squared penalty weights. To clean up the notation and optimize computational performance, we introduce a new diagonal matrix $P_k$ defined as:
$$P_k = W_k^{-2}$$
Since our original weight definition was $(W_k)_{ii} = \frac{1}{\sqrt{\vert{}(w_k)_i\vert{}}}$, squaring it yields $(W_k^2)_{ii} = \frac{1}{\vert{}(w_k)_i\vert{}}$. Taking the inverse to find $P_k$ simply flips the fraction, meaning the elements of $P_k$ track the direct magnitudes of the weights:
$$(P_k)_{ii} = \vert{}(w_k)_i\vert{}$$

###### Step 4: Final Dual Formulation
Substituting $P_k$ back into our derived equation yields the final optimized system:
$$w_{k+1} = P_k X^T (X P_k X^T + 2\lambda I)^{-1} y$$
This final expression is mathematically identical to the primal setup but only requires the inversion of an $M \times M$ matrix. By avoiding the $N \times N$ inversion completely, the per-iteration complexity collapses from $\mathcal{O}(N^3)$ to $\mathcal{O}(M^3)$, allowing wide network architectures to be trained efficiently.


## Convergence Analysis 
In general terms, the IRLS algorithm seeks to minimize a robust cost function defined as:
$$C_h(x) = \sum_{i=1}^k h(f_i(x))$$
where $f_i$ is a function defined on a domain taking non-negative real values (typically a distance or error function), and $h$ is the function chosen to make the result robust.

The algorithm **does not minimize $C_h(x)$ directly**. Starting from an initial value $x^0$, the IRLS iteratively solves a sequence of weighted least squares problems:
$$x^{t+1} = \arg\min_y \sum_{i=1}^k w_i(x^t) f_i(y)^2$$
where $w_i(x^t)$ is a weighting function.

To make the procedure converge to the minimum of the original function, the basic condition is that the weights are calculated according to the following formula:
$$w_i(x) = \frac{h'(f_i(x))}{2f_i(x)}$$
However, simply choosing these weights **does not intrinsically ensure that the iterations converge**It is necessary to mathematically demonstrate that each iteration of the IRLS actually reduces the overall cost $C_h(x)$.
#### Lemma 2.1

> **Lemma 2.1:** Let $g(x)$ be a concave function defined on a subset $D$ of real numbers, and let $g^s(c_i)$ be a supergradient evaluated at $c_i$. Let $c_i$ and $d_i$ in $D$ satisfy the inequality: $$\sum_{i=1}^k d_i g^s(c_i) \le \sum_{i=1}^k c_i g^s(c_i)$$Then it follows that: $$\sum_{i=1}^k g(d_i) \le \sum_{i=1}^k g(c_i)$$If the first inequality is strict, so is the second.

This lemma proves that the minimization of a weighted linear approximation guarantees the decrease of the original concave function $g(x)$.

To apply Lemma 2.1 to the IRLS context, it is necessary **to map the variables of the lemma to the variables of the minimization system**:

- The evaluation points are defined as the squares of the error function: $c_i = f_i(x^t)^2$ and $d_i = f_i(x^{t+1})^2$.
- Since the original cost function to minimize is $h(f_i(x))$, the concave function $g(x)$ of the lemma cannot correspond directly to $h(x)$. Because the input $c_i$ is a squared value, the function $g$ must first extract the square root and then apply $h$.
- Consequently, the lemma's function is defined as $g(x) = h(\sqrt{x})$.

By applying the substitution $g(x) = h(\sqrt{x})$ to Lemma 2.1, the exact condition for the validity of the IRLS algorithm is obtained.

#### Lemma 2.2
> **Lemma 2.2:** Let $h(\sqrt{x})$ be a concave function for $x \ge 0$, and let $x^t$ and $x^{t+1}$ be two values s.t.:$$\sum_{i=1}^k w_i^t f_i(x^{t+1})^2 \le \sum_{i=1}^k w_i^t f_i(x^t)^2$$where the weights $w_i^t$ are defined as in the weight equation. Then: $$\sum_{i=1}^k h(f_i(x^{t+1})) \le \sum_{i=1}^k h(f_i(x^t))$$ If the first inequality is strict, so is the second.

In summary, for the iterative minimization of squares to imply the minimization of the robust cost, the composition $h(\sqrt{x})$ must be concave. To guarantee the convergence of the sequence to the critical points, $h(\sqrt{x})$ must also be differentiable for $x \ge 0$.

#### The Problem with the $\ell_1$ Norm (Lasso)
Applying the IRLS framework to induce sparsity ([[L1-norm Regularization (Lasso)|L1 regularization]]), **the desired robust cost function is** $h(x) = x$. The variable of the problem is the network parameter $w_i$, and the function $f_i$ corresponds to the identity: $f_i(w) = \vert{}w_i\vert{}$.

Subjecting this formulation to the requirements of Lemma 2.2:
1. The function to be analyzed becomes $h(\sqrt{x}) = \sqrt{x}$.
2. The function $\sqrt{x}$ is concave.
3. However, the function $\sqrt{x}$ is not differentiable at $x=0$.

Consequently, **the weight $w(x)$ is not defined at that point**. If during the optimization a parameter reaches zero ($w_i = 0$), a mathematical singularity is generated. For this reason, the algorithm in its standard form cannot be guaranteed to converge to the minimum and presents a systematic risk of getting stuck at points where $f_i(x) = 0$.

#### Threshold $\epsilon$ and Huber Loss Induction
To resolve the singularity without precluding the achievement of sparsity, the practical implementation **modifies the calculation of the weights by introducing a constant $\epsilon > 0$ in the denominator**:
$$W_i = \frac{1}{\max(\vert{}w_i\vert{}, \epsilon)}$$
This perturbation acts not only as a numerical safeguard but strictly alters the cost function $h_{eff}$ that the algorithm is globally minimizing. The effective function can be derived by setting the equality with the theoretical definition of the weights:

$$\frac{h_{eff}'(w_i)}{w_i} = \frac{1}{\max(\vert{}w_i\vert{}, \epsilon)} \implies h_{eff}'(w_i) = \frac{w_i}{\max(\vert{}w_i\vert{}, \epsilon)}$$

By analytically integrating this derivative with respect to $w_i$, the domain splits into two sections:

**Regime 1: $\vert{}w_i\vert{} > \epsilon$ (Away from the origin)**
The derivative is $h_{eff}'(w_i) = \text{sgn}(w_i)$. The integration yields a linear behavior ($\ell_1$ norm):
$$h_{eff}(w_i) = \vert{}w_i\vert{} - C$$
**Regime 2: $\vert{}w_i\vert{} \le \epsilon$ (Around the origin)**
The derivative is $h_{eff}'(w_i) = \frac{w_i}{\epsilon}$. The integration yields a quadratic form:
$$h_{eff}(w_i) = \frac{w_i^2}{2\epsilon}$$
By matching the integration constants to impose continuity at $\vert{}w_i\vert{} = \epsilon$, the effective surrogate function becomes:
$$h_{eff}(w_i) = \begin{cases} \frac{w_i^2}{2\epsilon} & \text{for } \vert{}w_i\vert{} \le \epsilon \\ \vert{}w_i\vert{} - \frac{\epsilon}{2} & \text{for } \vert{}w_i\vert{} > \epsilon \end{cases}$$

**Theoretical Conclusions** The derivation demonstrates that the introduction of $\epsilon$ converts the problem from the minimization of the pure $\ell_1$ norm to the minimization of a Huber-type function. In the case of the Huber function, $h(x)$ is differentiable for $x \ge 0$ and the weights are well-defined everywhere. Since the new composition $h_{eff}(\sqrt{x})$ is strictly concave and everywhere differentiable, the formal requirements of Lemma 2.2 are fully met, thus ensuring the convergence of the algorithm to the minimum.

# References
- [[Iterative_Reweighted_Least_Squares.pdf]]
- [[Converge_IRLS.pdf]]