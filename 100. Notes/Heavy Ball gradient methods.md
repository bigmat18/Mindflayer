---
Data: 2026-04-04T20:42:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Smooth Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# “Poorman’s conjugate gradient”: Heavy Ball Gradient
Conjugate gradient and Quasi-Newton methods can be complex and expensive. There is a simpler alternative, known as the **Heavy Ball Gradient** (often simply called "Momentum" in Machine Learning).

The update process is slightly different from the usual:$$x^{i+1} \leftarrow x^i - \alpha^i \nabla f(x^i) + \beta^i (x^i - x^{i-1})$$

The term $\beta^i (x^i - x^{i-1})$ is the **"momentum term"**. The physical idea is that the point $x^i$ is "heavy" and tends to continue moving in the same direction it was already moving.
- At the same time, the gradient "force" $-\nabla f(x^i)$ steers the trajectory, pushing it toward the minimum $x_*$.
- A large "momentum" $\beta^i$ means less "zig-zagging" and a smoother trajectory.

Unlike the standard gradient, it is difficult to guarantee that $f(x^{i+1}) < f(x^i)$ at each step: in fact, **it is not a descent algorithm for $f$**. However, by appropriately choosing constant $\alpha^i$ and $\beta^i$, it behaves like a **linear descent algorithm for the distance $d$** from the optimum:
$$d^{i+1} = ||x^{i+1} - x_*|| \approx \le r ||x^i - x_*|| = d^i \quad \text{with} \quad r = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$$

This is the optimal achievable rate. To understand its impact: if the condition number is $\kappa = 1000$, the normal gradient has $r \approx 0.996$, while the Heavy Ball has $r \approx 0.938$. After 100 iterations, the gradient error only drops to $0.996^{100} = 0.6698$, while the Heavy Ball slashes it to $0.938^{100} = 0.0016$. A massive difference in practice!

## Mathematical Analysis I: The Two-Term Recurrence
To rigorously prove this convergence rate, we start from the definition of the recurrence, which in the Heavy Ball depends on two previous terms ($x^i$ and $x^{i-1}$). We can write it in block matrix form:

$$\begin{bmatrix} x^{i+1} - x_* \\ x^i - x_* \end{bmatrix} = \begin{bmatrix} x^i + \beta^i(x^i - x^{i-1}) - \alpha^i(\nabla f(x^i) - \nabla f(x_*)) - x_* \\ x^i - x_* \end{bmatrix}$$

(Note: we subtracted and added $\nabla f(x)$ which is $0$ at the optimum)

Applying the Mean Value Theorem to the gradient, we know there exists a point $w^i \in [x_*, x^i]$ such that $\nabla f(x^i) - \nabla f(x_*) = \nabla^2 f(w^i)(x^i - x_*)$. Substituting this term:

$$= \begin{bmatrix} (x^i - x_*) - \alpha^i \nabla^2 f(w^i)(x^i - x_*) + \beta^i(x^i - x^{i-1}) \\ x^i - x_* \end{bmatrix}$$

Grouping the terms to isolate $(x^i - x_*)$ and adding/subtracting $\beta^i x_*$:

$$= \begin{bmatrix} [I - \alpha^i \nabla^2 f(w^i)](x^i - x_*) + \beta^i(x^i - x^{i-1}) + \beta^i x_* - \beta^i x_* \\ x^i - x_* \end{bmatrix}$$

$$= \begin{bmatrix} [I - \alpha^i \nabla^2 f(w^i) + \beta^i I](x^i - x_*) - \beta^i(x^{i-1} - x_*) \\ x^i - x_* \end{bmatrix}$$

Finally, we extract the iteration matrix $C^i$:

$$= \begin{bmatrix} (1 + \beta^i)I - \alpha^i \nabla^2 f(w^i) & -\beta^i I \\ I & 0 \end{bmatrix} \begin{bmatrix} x^i - x_* \\ x^{i-1} - x_* \end{bmatrix}$$

If we could find $\alpha^i$ and $\beta^i$ such that the norm of this matrix $||C^i|| < 1$, we would have proven linear convergence. Unfortunately, it is not that simple, because $||C^i|| > 1$.

## Mathematical Analysis II: Spectral Radius (Complicated)
Since $C^i$ is not symmetric, its norm is greater than or equal to its **spectral radius** $\rho(C^i) = \max_j \{|\lambda_j(C^i)|\}$ (where the eigenvalues can be complex, so $|\cdot|$ is the modulus, not the normal absolute value).

Through a complex block diagonalization, the spectral radius $\rho(C^i)$ is decomposed into the maximum of the spectral radii of $n$ $2 \times 2$ submatrices:

$$C_j = \begin{bmatrix} 1 + \beta^i - \alpha^i \lambda_j(D) & -\beta^i \\ 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

_(where $\lambda_j(D)$ are the eigenvalues of the Hessian)_.

Solving the characteristic polynomial of these submatrices (an extremely tedious process) yields an upper bound for the spectral radius:

$$\rho(C^i) \le \sqrt{\beta^i} = \max\{|1 - \sqrt{\alpha^i \tau}|, |1 - \sqrt{\alpha^i L}|\}$$

To minimize this maximum, the **optimal $\alpha$** turns out to be:

$$\alpha = \frac{4}{(\sqrt{L} + \sqrt{\tau})^2} \implies \sqrt{\beta} = \frac{\sqrt{L} - \sqrt{\tau}}{\sqrt{L} + \sqrt{\tau}} < 1$$

This gives us exactly the optimal convergence rate $r = \sqrt{\beta} = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$. This would hold true if we could prove linear convergence directly with $r = \sqrt{\beta}$, which is _almost_ true, but not entirely.

## Mathematical Analysis III: Gelfand's Formula (++Complicated)
Let's make a simplifying assumption: suppose $f$ is quadratic. This makes the Hessian $\nabla^2 f$ constant, and consequently the iteration matrix $C^i = C$ is constant.

By recursion, the error at step $i$ is bounded by:

$$||E^i|| \le ||C^i|| \cdot ||E^0||$$

_(where $C^i$ is the matrix raised to the $i$-th power)_.

Here **Gelfand's Formula** comes into play:

$$\rho(C) = \lim_{i \to \infty} ||C^i||^{1/i}$$
This mathematically implies that:

$$\forall \epsilon > 0 \quad \exists h \quad \text{s.t.} \quad ||C^i|| \le (\rho(C) + \epsilon)^i \quad \forall i \ge h$$

_What does this mean in practice?_ It means that the error can grow or oscillate at the beginning, but if we wait for a sufficient number of iterations $h$ ("large"), sooner or later the algorithm **"starts to converge"** with a nearly linear rate dictated by the spectral radius $\rho(C)$.

In case the function $f$ is **non-convex**, the algorithm still converges if $\beta \in [0, 1)$ and $\alpha \in (0, 2(1-\beta)/L)$, although the window for choosing $\alpha$ becomes very narrow as $\beta \to 1$.


## What if $\tau = 0$? (Accelerated Gradient Method)

What happens if the function is not strongly convex ($\tau = 0$)?

In this case, the Heavy Ball only guarantees an error rate of $O(1/i)$, which is theoretically no better than the standard gradient.

To overcome this theoretical limit, a variant called **Accelerated Gradient** (often known as Nesterov's Method) is used, whose pseudocode hides genuine mathematical "black magic":

Plaintext

```
procedure y = ACG(f, x, \epsilon)
    x_- <- x; \gamma <- 1;
    do { // warning: black magic ahead
        \gamma_- <- \gamma; 
        \gamma <- (\sqrt{4\gamma_-^2 + \gamma_-^4} - \gamma_-^2)/2; 
        \beta <- \gamma(1/\gamma_- - 1);
        y <- x + \beta(x - x_-); 
        g <- \nabla f(y); 
        x_- <- x; 
        x <- y - (1/L)g;
    } while( ||g|| > \epsilon );
```

_Explanation of the differences:_ The ACG is very similar to the Heavy Ball, but with a crucial difference: **the gradient is evaluated at the "predicted" point by the momentum ($y$), and not at the current point ($x$)**.

This small theoretical change ensures the **optimal** possible convergence rate for merely L-smooth functions: $O(LD^2/\sqrt{\epsilon})$. If the function is also $\tau$-convex, it achieves the same optimal linear rate as the Heavy Ball.

However, in practice the ACG is consistently a bit slow ("slowish"): it was carefully designed to mathematically guarantee the best possible behavior in the _worst case_ (worst-case behaviour), and it behaves exactly as it was programmed to do.
# References