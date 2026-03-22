---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# Gradient Method with Fixed Stepsize

When performing optimization, an alternative to spending computational resources dynamically finding the step size at each iteration (like in Exact or Inexact Line Search) is to use an "Extremely inexact Line Search": a **Fixed Stepsize** strategy.

By setting $\alpha^i = \overline{\alpha}$ for every iteration, the algorithm becomes very simple and inexpensive per step, but also very rigid. It's a "one size fits all" choice, so the value of $\overline{\alpha}$ must be chosen very carefully.

## The Problem with Normalized Directions

For a sequence $\{x^i\}$ to converge to a finite point $x_*$, the distance between consecutive points must go to zero:

$$\{||x^{i+1} - x^i|| = \alpha^i ||d^i||\} \to 0$$

If we were to use a _normalized_ gradient as our descent direction, meaning $d^i = -\nabla f(x^i)/||\nabla f(x^i)||$, then the length of the direction vector would always be $||d^i|| [cite_start]= 1$. For the step distance to reach zero, the step size $\alpha^i$ itself would have to shrink to zero ($\alpha^i \to 0$), which is impossible if we are using a fixed stepsize $\overline{\alpha} > 0$.

Luckily, by using the standard, unnormalized anti-gradient $d^i = -\nabla f(x^i)$, we avoid this problem. The condition becomes:

$$\{||x^{i+1} - x^i||\} \to 0 \iff \{\nabla f(x^i)\} \to 0$$

This is precisely what we want: the physical steps naturally become smaller and shrink to zero as we approach a stationary point (where the gradient vanishes).

## L-Smoothness and The Crucial Bound

To guarantee that the fixed stepsize will actually decrease the function value at each iteration ($f(x^{i+1}) < f(x^i)$), we must bound how rapidly the gradient can change. This is where **L-smoothness** is critical.

If the function $f$ is L-smooth, then our 1D line-search function $\varphi$ is $[L||d||^2]$-smooth. Knowing that $d = -\nabla f(x)$, we can evaluate the derivative of $\varphi$ at $\alpha = 0$:

$$\varphi'(0) = -||\nabla f(x)||^2 = -||d||^2$$

Because $\varphi'$ cannot change faster than $L||d||^2$, we can establish a strict upper bound on how the slope evolves as we take a step $\alpha$:

$$\varphi'(\alpha) \le \varphi'(0) + L||d||^2 \alpha = ||\nabla f(x)||^2 (L\alpha - 1)$$

By observing the term $(L\alpha - 1)$, we can see that as long as $L\alpha - 1 < 0$, the slope $\varphi'(\alpha)$ is guaranteed to be negative, meaning the function is still going down. Therefore:

$$\varphi'(\alpha) \le 0 \quad \forall \alpha \in [0, \overline{\alpha} = 1/L)$$

This suggests a safe, proposed fixed stepsize of $\overline{\alpha} = 1/L$.

Integrating this worst-case linear bound gives us the immediate, guaranteed estimate of the error decrease at each step:

$$f(x^{i+1}) - f(x^i) \le -||\nabla f(x^i)||^2/2L$$

**The Bad News for General L-Smooth Functions:** If we define the error as $a^i = f(x^i) - f_*$, we get $a^{i+1} \le a^i - \Delta^i$. Because we are subtracting a varying $\Delta^i$ rather than multiplying by a fraction $r < 1$, this results in **sublinear convergence**. It can be proven that the iterations scale as $O(LD^2/\epsilon)$, meaning reaching high precision is very slow. Algorithms can only go so far with "nasty" problems.

---

## Mathematically Speaking: Eigenvalues & Matrix Norms

To understand how to make this algorithm exponentially faster, we must look at the geometry of the space, governed by matrix norms and eigenvalues.

When we multiply a vector $x$ by a matrix $Q$ (a linear mapping $y = Qx$), the matrix stretches and rotates the vector. The **Matrix norm induced by a vector norm** measures the _maximum_ possible stretching that $Q$ can apply to any vector:

$$||Q||_p = \max\{||Qx||_p : ||x|| = 1\} \equiv \max\{||Qx||/||x|| : x \ne 0\}$$

This definition gives us a crucial mathematical inequality: the length of the transformed vector is always bounded by the norm of the matrix times the length of the original vector:

$$||Qx||_p \le ||Q||_p ||x||_p \quad \forall x \in \mathbb{R}^n$$

**Variational Characterization:** The spectral norm (the $L_2$ norm) of a matrix is strictly related to its eigenvalues. Let's look at the Eigenvalue/Eigenvector Pairs (eep) where $Qv = \lambda v$:

- If $(\lambda, v)$ is an eep for $Q$, then $(c\lambda, v)$ is an eep for $cQ$ (scaling the matrix scales the eigenvalues).
    
- $(1+\lambda, v)$ is an eep for $I+Q$ (adding the Identity matrix shifts all eigenvalues by +1).
    
- $(\lambda^2, v)$ is an eep for $Q^2 = QQ$.
    

Because of these properties, if $Q$ is a **symmetric matrix** (like the Hessian $\nabla^2 f$), its norm is simply the maximum absolute eigenvalue:

$$||Q|| = \max\{|\lambda_1(Q)|, |\lambda_n(Q)|\}$$

_(Note: $\lambda_1$ represents the largest algebraic eigenvalue, and $\lambda_n$ the smallest)._

---

## Convergence Rate with Strong Convexity ($\tau$-convex)

Things change drastically if we add the assumption of **Strong Convexity** ($\tau$-convexity). We want to prove that with a proper choice of $\alpha$, the geometric distance to the optimum $x_*$ decreases "fast".

Let's define our next point $z = x - \alpha\nabla f(x)$. We want to measure its distance to the optimum $x_*$. Since the gradient at the optimum is zero ($\nabla f(x_*) = 0$), we can add it to our equation for free:

$$z - x_* = x - \alpha\nabla f(x) - x_* + \alpha\nabla f(x_*)$$

Grouping the terms, we get:

$$= (x - x_*) - \alpha(\nabla f(x) - \nabla f(x_*))$$

By applying the **Mean Value Theorem** on the gradient $\nabla f$, we know there exists some intermediate point $w \in [x, x_*]$ such that the difference in gradients is exactly the Hessian evaluated at $w$ multiplied by the distance vector:

$$\nabla f(x) - \nabla f(x_*) = \nabla^2 f(w)(x - x_*)$$

Substituting this back into our equation, we factor out $(x - x_*)$:

$$z - x_* = (x - x_*) - \alpha\nabla^2 f(w)(x - x_*) = (I - \alpha\nabla^2 f(w))(x - x_*)$$

Now, applying the matrix norm inequality we learned earlier, we can bound the distance:

$$||z - x_*|| \le ||I - \alpha\nabla^2 f(w)|| ||x - x_*||$$

To make the distance shrink as fast as possible, our goal is to **minimize the matrix norm** $r = ||I - \alpha\nabla^2 f(w)||$.

---

## Mathematically Speaking: The Choice of $\alpha$

We need to minimize the norm $r$. Using the variational characterization rules from earlier, we know that the eigenvalues of $I - \alpha \nabla^2 f(w)$ are exactly $1 - \alpha \lambda_i$. Because the matrix is symmetric, its norm is the maximum absolute eigenvalue:

$$r = ||I - \alpha\nabla^2 f(w)|| = \max\{|1 - \alpha\lambda_1(\nabla^2 f(w))|, |1 - \alpha\lambda_n(\nabla^2 f(w))|\} < 1$$

To make this maximum as small as possible, we must choose $\alpha$.

- If $\alpha$ is too small, $1 - \alpha\lambda_n \approx 1$, meaning no progress.
    
- If $\alpha$ is too large, $|1 - \alpha\lambda_1|$ becomes a large negative number, causing oscillations or divergence.
    

The optimal $\alpha$ must be perfectly balanced such that the smallest eigenvalue term is strictly positive ($1 - \alpha\lambda_n > 0$) and the largest eigenvalue term is strictly negative ($1 - \alpha\lambda_1 < 0$). Since the second term is negative, its absolute value is its negation:

$$r = \max\{-1 + \alpha\lambda_1, 1 - \alpha\lambda_n\}$$

We generally don't know the exact eigenvalues $\lambda_1$ and $\lambda_n$, but we know they are bounded by $L$ (maximum curvature) and $\tau$ (minimum curvature, due to strong convexity). Therefore, we can bound our rate $r \le \overline{r}$:

$$\overline{r} = \max\{-1 + \alpha L, 1 - \alpha \tau\}$$

To minimize this maximum, we look for the point where the two functions intersect (because if one term goes up, the other goes down):

$$-1 + \alpha L = 1 - \alpha \tau \Rightarrow \alpha(L + \tau) = 2 \Rightarrow \alpha = \frac{2}{L + \tau}$$

Substituting this optimally balanced fixed stepsize back into the rate gives us our final convergence factor:

$$\overline{r} = \frac{L - \tau}{L + \tau} = \frac{\overline{\kappa} - 1}{\overline{\kappa} + 1} < 1$$

_(where $\overline{\kappa} = L/\tau \ge 1$ is the worst-case condition number)_.

**Conclusion:** With $\alpha = 2/(L+\tau)$, the algorithm achieves **linear convergence**:

$$||x^{k+1} - x_*|| \le r^k ||x^1 - x_*||$$

A small difference in the mathematical properties of $f$ (adding strong convexity) makes a massive difference in the convergence behavior—proving that the properties of the function are often more important than the algorithm itself!

---

Would you like me to apply this same level of mathematical breakdown to the next section on "Twisted Gradient Methods" and "Newton's Method"?
# References