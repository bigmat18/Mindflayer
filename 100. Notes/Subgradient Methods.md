---
Data: 2026-05-05T00:39:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Subgradient Methods

This section dives into the algorithmic construction of **Subgradient Methods**, which are the fundamental tools used to minimize nondifferentiable convex functions. Since we cannot guarantee that moving opposite to a subgradient strictly decreases the function value, we must rely on a different geometric property to ensure progress.

### The Fundamental Relationship: Getting Closer to the Optimum
As established previously, any negative subgradient ($-g$) at a point $x^i$ points "towards" the global minimum $x_*$. This means that taking an appropriate step along $-g$ will bring the next iterate $x^{i+1}$ physically closer to $x_*$, even if the function value $f(x^{i+1})$ temporarily increases.
#### The Update Rule
The algorithm's update rule is defined as:
$$x^{i+1} = x^i - \alpha^i g^i$$
- **$x^{i+1}$**: The next position.
- **$x^i$**: The current position.
- **$\alpha^i > 0$**: The stepsize (length of the movement).
- **$g^i \in \partial f(x^i)$**: The subgradient at the current point.

#### Mathematical Derivation
To prove the algorithm converges, we analyze the **squared Euclidean distance** to the optimum, $\|x^{i+1} - x_*\|^2$. We expand this using the update rule:
$$\|x^{i+1} - x_*\|^2 = \|x^i - \alpha^i g^i - x_*\|^2$$
$$\|x^{i+1} - x_*\|^2 = \|x^i - x_*\|^2 + 2\alpha^i \langle g^i, x_* - x^i \rangle + (\alpha^i)^2 \|g^i\|^2$$

Using the **Subgradient Inequality**, which states that for convex functions $f(x_*) \ge f(x^i) + \langle g^i, x_* - x^i \rangle$, we can rearrange it as:
$$\langle g^i, x_* - x^i \rangle \le f(x_*) - f(x^i)$$
Substituting this into our expansion, we obtain the **Fundamental Relationship**:

$$\|x^{i+1} - x_*\|^2 \le \|x^i - x_*\|^2 + \underbrace{2\alpha^i (f_* - f(x^i))}_{\text{Negative Term (< 0)}} + \underbrace{(\alpha^i)^2 \|g^i\|^2}_{\text{Positive Term (> 0)}}$$
This inequality represents a "tug-of-war" between two forces:

1. **The Negative Term ($2\alpha^i (f_* - f(x^i))$)**: Since $f_*$ is the global minimum, $(f_* - f(x^i))$ is always $\le 0$. This term pulls the distance down, moving us closer to the optimum. It depends **linearly** on $\alpha^i$.

2. **The Positive Term ($(\alpha^i)^2 \|g^i\|^2$):** This is the "noise" or error caused by the non-smoothness of the function. It pushes the distance up. It depends **quadratically** on $(\alpha^i)^2$.

**The Core Logic:**
Because the negative term is linear ($\alpha$) and the positive term is quadratic ($\alpha^2$), for a **sufficiently small stepsize** $\alpha > 0$, the linear benefit will always outweigh the quadratic error.
- **Result:** $\|x^{i+1} - x_*\|^2 < \|x^i - x_*\|^2$, meaning the new point $x^{i+1}$ is strictly closer to $x_*$ than $x^i$ was.

A common concern is that the formula uses $x_*$ and $f_*$, which are unknown (the very values we are trying to find).

- **Theoretical Guarantee:** This analysis is a **convergence proof**. It demonstrates that even if we don't know where $x_*$ is, the algorithm is mathematically "forced" to get closer to it as long as the stepsize is managed correctly.
- **The Stepsize Strategy:** Since we cannot perfectly balance the terms without knowing $f_*$, we typically use a **Diminishing Stepsize** (e.g., $\alpha^i = 1/i$). This ensures that as iterations progress, the error term ($\alpha^2$) vanishes faster than the progress term ($\alpha$), guaranteeing convergence to the exact minimum.
- **"Best-so-far" Tracking:** Because $f(x^i)$ might not decrease at every single step, practitioners always keep track of the best value encountered during the process:
$$f_{best}^i = \min_{j=0 \dots i} f(x^j)$$

### Stepsize Strategy 1: [[Diminishing-Square Summable (DSS)]]

### Stepsize Strategy 2: [[Polyak Stepsize]]

### Stepsize Strategy 3: [[Target Level Stepsize]]

### Deflected Subgradient Methods
"Want a better direction? Use a better model!" Since moving strictly along the negative subgradient causes erratic zig-zagging, we can "deflect" the direction using momentum from previous steps (similar to conjugate gradient methods for smooth functions).

The new direction $d^i$ is a combination of the current subgradient and the previous direction:
$$d^i = \gamma^i g^i + (1 - \gamma^i) d^{i-1}$$
$$x^{i+1} = x^i - \alpha^i d^i$$

To maintain theoretical convergence, strict and sometimes "funny" rules are needed for the mixing parameter $\gamma^i$ and stepsize $\alpha^i$:
- **Stepsize-restricted (Polyak approach):** $\alpha^i = \beta^i (f^i - f_*) / ||d^i||^2$ with the restriction $\beta^i \le \gamma^i$. As deflection increases, the stepsize must decrease.
- **Deflection-restricted (DSS approach):** A complex formula restricts $\gamma^i$ based on the previous step size and error: $\frac{\alpha^{i-1}||d^{i-1}||^2}{(f^i - f_*) + \alpha^{i-1}||d^{i-1}||^2} \le \gamma^i$. As $f(x^i) \rightarrow f_*$, the allowed deflection goes down.

![[Pasted image 20260510153534.png | 350]]

Alternatively, $\gamma^i$ can be found via a closed-formula projection: $\gamma^i \in \text{argmin} \{ ||\gamma g^i + (1-\gamma) d^{i-1}||^2 : \gamma \in [0,1] \}$. While deflection does help stabilize the path in practice, the improvement is incremental and "not much" in the grand scheme of nonsmooth optimization limits.
# References