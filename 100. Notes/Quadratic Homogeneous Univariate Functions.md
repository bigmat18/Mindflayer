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
# Quadratic Homogeneous Univariate Functions

The next simplest functions are homogeneous quadratics: $f(x) = a x^2$, where the fixed parameter $a \in \mathbb{R}$ is the **quadratic coefficient** (or curvature). Again, a bijection with $\mathbb{R}$.

The function is always symmetric around $x=0$ ($f(x) = f(-x)$), but its monotonicity depends on the sign of $a$ and the quadrant.

*   **Convex Case ($a > 0$):**
    *   Decreasing for $x \le 0$: As $x$ moves left from 0, $f(x)$ decreases to 0.
    *   Increasing for $x \ge 0$: As $x$ moves right from 0, $f(x)$ increases.
    *   Shape: Parabola opening upward, global minimum at $x=0$.
![[Pasted image 20260217190455.png | 400]]

*   **Constant Case ($a = 0$):** 
    *   Non-increasing for $x \le 0$ and non-decreasing for $x \ge 0$ (constant at 0).
![[Pasted image 20260217190609.png | 400]]

*   **Concave Case ($a < 0$):**
    *   Increasing for $x \le 0$: As $x$ moves left from 0, $f(x)$ increases (becomes less negative).
    *   Decreasing for $x \ge 0$: As $x$ moves right from 0, $f(x)$ increases (goes to $-\infty$).
    *   Shape: Parabola opening downward, global maximum at $x=0$, unbounded below.
![[Pasted image 20260217190557.png | 400]]
    
The larger $|a|$, the steeper the parabola (curvature effect).
### Optimization
Unconstrained optimization:
*   If $a > 0$: $\min = \text{argmin} = 0$, $\max = +\infty$.
*   If $a < 0$: $\max = \text{argmax} = 0$, $\min = -\infty$.
*   Symmetric around 0: $\text{argmin/argmax} = \pm \infty$ for the unbounded directions.

The more interesting case is the **box-constrained** optimization: $\min \{ f(x) : x \in [x_-, x_+] \}$.

- **Case $a > 0$ (Convex):** Three subcases based on the position of 0 relative to the interval.
    1.  If $x_+ < 0$ (interval left of 0): Decreasing on $X$, so $\text{argmin} = x_+$, $\text{argmax} = x_-$.
    2.  If $x_- > 0$ (interval right of 0): Increasing on $X$, so $\text{argmin} = x_-$, $\text{argmax} = x_+$.
    3.  If $x_- \le 0 \le x_+$ (interval contains 0): $\text{argmin} = 0$ (global minimum). For the maximum (since the function increases away from 0), $\text{argmax} = \text{argmax} \{ f(x_-), f(x_+) \}$.
    
    Works for infinite bounds: $a \cdot (\pm \infty)^2 = +\infty$ (not $-\infty$).

- **Case** $a < 0$, the roles of min/max reverse (unbounded directions yield $-\infty$).
- **Case** $a = 0$, constant (min=max=0 everywhere).


**Closed-Form Solution:** Again $O(1)$ with direct formulas. Note that $\max \{ f(x) \}$ and $\min \{ f(x) \}$ are not always symmetric (cf. the $a > 0$ case where the max requires testing endpoints).

# References