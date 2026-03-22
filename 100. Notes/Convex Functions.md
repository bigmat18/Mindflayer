---
Data: 2026-03-22T17:21:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Unconstrained Multivariate Optimality and Convexity]]"
Area: "[[Master's degree]]"
---
# A Quick Look to Convex Functions
As we've seen, finding a global minimum in multiple dimensions is extremely difficult unless the function has a specific, highly desirable property: **convexity**. Convex functions are the ideal class for optimization because they eliminate the distinction between local and global minima.

## Convex functions
The fundamental mathematical definition of a convex function in multiple dimensions is based on linear interpolation between any two points in its domain:
$$f \text{ convex} \equiv \forall x, z \in \mathbb{R}^n, \alpha \in [0,1]$$
![[Pasted image 20260322174118.png | 350]]

$$\alpha f(x) + (1-\alpha)f(z) \ge f(\alpha x + (1-\alpha)z)$$
![[Pasted image 20260322174159.png | 350]]

**Explanation**: Geometrically, if you pick any two points $x$ and $z$, the value of the function at their intermediate point (any convex combination $\alpha x + (1-\alpha)z$) will always be less than or equal to the value of the straight line segment (the secant) connecting $f(x)$ and $f(z)$ at that same point.


Several immediate properties derive from this definition:
- **Convexity does not imply differentiability:** A convex function is not necessarily $C^1$. For example, the $L_1$ norm $f(x) = ||x||_1$ is convex everywhere but has non-differentiable points (sharp "kinks" at the origin).
![[Pasted image 20260322174235.png | 350]]

- **Concave functions:** $f \text{ concave} \equiv -f \text{ convex}$.
![[Pasted image 20260322174323.png | 350]]

- **Unboundedness:** For a convex function, $\max\{f(x) : x \in \mathbb{R}^n\} = +\infty$ (unless it is a constant function $f(x)=c$).
- **Quadratic functions:** A quadratic function $f(x) = \frac{1}{2}x^T Q x + qx$ is convex if and only if $Q \ge 0$ (the matrix $Q$ is positive semi-definite). Exactly the opposite is true for a concave function ($Q \le 0$).
- As a great man said, *"(convex) optimization is a one-sided world"*.
- **Only $f$ both convex and concave:** The only function that satisfies both properties simultaneously is a linear (affine) function. (Explanation: proving this requires showing that the inequality becomes a strict equality, which perfectly defines an affine form $f(x) = \langle b,x \rangle + c$)


If a convex function is smooth (differentiable), we get incredibly powerful mathematical guarantees linking its shape to its derivatives.
#### First-Order Conditions ($f \in C^1$)
If $f \in C^1$ is convex, two crucial properties hold:

1. **$\nabla f$ is monotone:** $$\langle \nabla f(z) - \nabla f(x), z - x \rangle \ge 0 \quad \forall x, z$$
   Explanation: In 1D, this is $(f'(z)-f'(x))(z-x) \ge 0$, meaning that $f'(z)-f'(x)$ and $z-x$ have the same sign. If $z \ge x \Rightarrow f'(z) \ge f'(x)$, indicating that the derivative is monotonically non-decreasing. The gradient points consistently outward.

2. **The first-order model is a global underestimator:** $$L_x(z) = f(x) + \langle \nabla f(x), z - x \rangle \le f(z)$$
![[Pasted image 20260322174509.png | 350]]

   Explanation: Geometrically, the epigraph of the function (everything above the curve, $epi(f)$) is fully contained by the half-space defined by the tangent plane ($epi(L_x) \supseteq epi(f)$).
   

From this derives the most important result of convex optimization: if we find a stationary point where the gradient is zero ($\nabla f(x) = 0$), the inequality simply becomes:
$$\nabla f(x) = 0 \Rightarrow f(z) \ge f(x) \quad \forall z \in \mathbb{R}^n$$
Therefore, $x$ is a global minimum.

![[Pasted image 20260322174558.png | 350]]

#### Second-Order Conditions ($f \in C^2$)
If $f \in C^2$, convexity is completely determined by the [[Hessians]] matrix:
$$f \text{ convex} \equiv \nabla^2 f(x) \ge 0 \quad \forall x \in \mathbb{R}^n$$
The absolute best case for optimization is when $f \in C^2$ with $\nabla^2 f \ge \tau I$ (where $\tau > 0$). This guarantees a minimum amount of strict curvature everywhere, making optimization algorithms exceptionally fast.


## Basic convex functions
Sometimes taking the second derivative to prove $\nabla^2 f(x) \ge 0$ is too hard. Instead, we can prove a function is convex by showing it belongs to a list of known foundational convex functions:

1. $f(x) = bx + c$ (affine) — both convex and concave.
2. $f(x) = \frac{1}{2}x^T Q x + qx$ (quadratic) convex.
3. $f(x) = e^{ax}$ for any $a \in \mathbb{R}$. *(Its derivative $ae^{ax}$ is strictly increasing).*
4. restricted to $x \ge 0$, $f(x) = -\ln(x)$. *(Its derivative $-1/x$ is negative increasing).*
5. restricted to $x \ge 0$, $f(x) = x^a$ for $a \ge 1$ or $a \le 0$. *(Note: only positive even integers $a$ make $x^a$ convex on all $\mathbb{R}$, as the second derivative $a(a-1)x^{a-2}$ is always positive).*
6. $f(x) = ||x||_p$ for $p \ge 1$.
7. $f(x) = \max\{x_1, ..., x_n\}$.
8. Given $Q \in \mathbb{R}^{n \times n}$ symmetric, with eigenvalues $\lambda_1 \ge \lambda_2 \ge ... \ge \lambda_n$: $f_m(Q) = \sum_{i=1}^m \lambda_i$ (the sum of the $m$ largest eigenvalues).

## Convexity-preserving operations
These basic functions can be combined using specific operations that guarantee the final result remains convex:

1. **Non-negative combination:** $f, g$ convex, $\delta, \beta \in \mathbb{R}_+ \Rightarrow \delta f + \beta g$ convex.
2. **Supremum:** $\{f_i\}_{i \in I}$ ($\infty$-ly many) convex functions $\Rightarrow f(x) = \sup_{i \in I}\{f_i(x)\}$ convex.
3. **Pre-composition with linear mapping:** $f \text{ convex} \Rightarrow f(Ax+b)$ convex.
4. **Post-composition:** $f: \mathbb{R}^n \rightarrow \mathbb{R}$ convex, $g: \mathbb{R} \rightarrow \mathbb{R}$ convex increasing $\Rightarrow g(f(x))$ convex.
5. **Infimal convolution:** $f_1, f_2 \text{ convex} \Rightarrow f(x) = \inf\{f_1(x_1) + f_2(x_2) : x_1 + x_2 = x\}$ convex.
6. **Value function of convex constrained problem:** $g \text{ convex} \Rightarrow f(x) = \inf\{g(z) : Az = x\}$ convex.
7. **Partial minimization:** $g(x,z): \mathbb{R}^{n+m} \rightarrow \mathbb{R} \text{ convex} \Rightarrow f(x) = \inf\{g(x,z) : z \in \mathbb{R}^m\}$ convex.
8. **Perspective or dilation function of $f$:** $f(x) \text{ convex} \Rightarrow p(x, u) = u f(x/u)$ convex on $u > 0$.

## Why convex and not unimodal?
In 1D ($n=1$), an $f$ **unimodal** function (one that goes down to a single minimum and then up) is sufficient for optimization. In multiple dimensions, the equivalent concept is **quasiconvexity**.

A function is quasiconvex if:
$$\alpha f(x) + (1-\alpha)f(z) \le \max\{f(x), f(z)\}$$

$f$ quasiconvex $\iff \forall l$ nonempty sublevel set $S(f,l) = \{x : f(x) \le l\}$ is a (possibly, infinite) interval (in fact a convex set).

While it is true that $f \text{ convex} \Rightarrow f \text{ quasiconvex}$, the reverse is not true. So why not build optimization around quasiconvexity?

**Issue: the algebra of quasiconvex (not convex) functions is "weaker"**.
- $f$ quasiconvex, $\delta \in \mathbb{R}_+ \Rightarrow \delta f$ quasiconvex is true.
- **But $f, g$ quasiconvex $\Rightarrow f+g$ quasiconvex is FALSE.**
  Explanation: Consider two quasiconvex "downward spike" functions, $s_{-1}(x) = \min\{|x+1|, 1\}$ and $s_1(x) = \min\{|x-1|, 1\}$. Their sum creates a function with two separate "dips", making the sublevel set $S(f,0) = \{-1, 1\}$, which is not an interval. Quasiconvexity is completely destroyed.

Because we cannot safely sum them, there is no (or much weaker) Disciplined QuasiConvex Programming. It is unlikely for $f$ to be "naturally" quasiconvex if it is built from complex operations. However, this does not mean it is impossible: in machine learning, Neural Networks (NN) are often empirically found to be $\approx$ quasiconvex, allowing local methods to succeed.

# References