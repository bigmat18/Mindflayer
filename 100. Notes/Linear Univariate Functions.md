---
Data: 2026-02-17T19:18:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Linear Univariate Functions

The simplest possible function is a linear one: $f(x) = bx$, where the fixed parameter $b \in \mathbb{R}$ is the **linear coefficient** (or slope). This establishes a bijection between $\mathbb{R}$ (slope) and the space of linear functions, meaning every real number corresponds to a unique linear function.

The behavior of $f(x)$ depends entirely on the sign of $b$, determining whether the function is increasing, decreasing, or constant.

*   **Increasing Case ($b > 0$):**
    The function is strictly increasing: $x > z \implies f(x) > f(z)$.
    The line slopes upward from left to right.
![[Pasted image 20260217185226.png | 400]]

*   **Non-Decreasing Case ($b = 0$):** 
    The function is non-decreasing (constant): $x > z \implies f(x) \ge f(z)$.
    It is a horizontal line at $y = 0$.
![[Pasted image 20260217185324.png | 400]]
	
*   **Decreasing Case ($b < 0$):** 
    The function is strictly decreasing: $x > z \implies f(x) < f(z)$.
    The line slopes downward from left to right.
![[Pasted image 20260217185358.png | 400]]

### Optimizing a linear function
Unconstrained optimization ($\min \{ f(x) : x \in \mathbb{R} \}$) is trivial:
*   If $b \ne 0$, then $\min = -\infty$ and $\max = +\infty$.
*   If $b = 0$, then $\min = \max = 0$.

Of greater interest is the **box-constrained** (or bounded) version:
$$(P) \quad \min \{ f(x) : x \in [x_-, x_+] \}$$
where $-\infty \le x_- \le x_+ \le +\infty$ defines possibly half-infinite intervals. Box constraints are very simple but often practically useful (e.g., physical limits on variables).

- **Case $b > 0$ (Increasing):**
    *   $\text{argmin} = x_-$ (minimum at left endpoint).
    *   $\min = f(x_-)$.
    *   $\text{argmax} = x_+$ (maximum at right endpoint).
    *   $\max = f(x_+)$.
    *   This works even for infinite bounds: $b \cdot (-\infty) = -\infty$ and $b \cdot (+\infty) = +\infty$.
- **Case** $b < 0$ **(decreasing)**, the roles reverse: $\text{argmin} = x_+$, $\text{argmax} = x_-$.
- **Case** $b = 0$, every point in $X = [x_-, x_+]$ is an optimal solution ($\min = \max = 0$).

**Closed-Form Solution:** The optimization has complexity $O(1)$ with a direct formula. Don't get used to it—solving simple problems is the basis for tackling more complex ones.

## Aside: Optimizing over an "Open" Box (Once and for All)
Could we use an open interval for the feasible set? Consider $X = (x_-, x_+) = \{ x \in \mathbb{R} : x_- < x < x_+ \}$.
###### Why It's a Bad Idea:
*   Similar to the asymptotic case, the **infimum exists**, but the **minimum does not**: $\inf \exists$ but $\min \nexists$.
*   We have a finite $f_*$ but no $x_*$ achieving it.
*   The feasible set being open means the boundaries are excluded, so even a bounded function may not attain its minimum if the optimum lies on the boundary.

Example: For an increasing linear function $f(x) = x$ on $(x_-, x_+)$, $f_* = x_-$ but $x_-$ is not in $X$.
###### Practical Considerations:
*   **Physical Interpretation:** Variables $x$ often represent physical quantities that cannot be measured or chosen with infinite precision (Planck scale, Heisenberg's Uncertainty Principle, etc.). Open intervals don't make sense in applications.
*   **Algorithmic Issues:** In theory, open sets pose problems for guaranteeing convergence. In practice, floating-point arithmetic handles this via $\varepsilon$-approximation, and plenty of $\varepsilon$-optimal solutions exist regardless.
*   **Workaround:** If boundaries "cannot be touched" (e.g., due to hardware limits), use a shrunken closed interval: $X = [x_- + \varepsilon_-, x_+ - \varepsilon_+]$ for small $\varepsilon_\pm > 0$.

**Just use closed intervals and be done with it.** This generalizes to "just use closed sets and be done with it" in higher dimensions. Closed feasible sets ensure compactness (when bounded), guaranteeing the existence of a minimum via the Extreme Value Theorem.
# References