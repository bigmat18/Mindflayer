---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# A Quick Look to Convex Functions

## The Fundamental Definition of Convexity

To understand why convex functions are the "holy grail" of optimization, we must start with their geometric and mathematical definition. A function $f$ is convex if, for any two points in its domain, the line segment connecting them lies entirely above (or on) the graph of the function itself. 

Mathematically, this means that for any $x, z \in \mathbb{R}^n$ and any scalar $\alpha \in [0,1]$, the following inequality must hold:
$$\alpha f(x) + (1-\alpha)f(z) \ge f(\alpha x + (1-\alpha)z)$$
- The left side of the inequality, $\alpha f(x) + (1-\alpha)f(z)$, represents the linear interpolation (the value on the secant line) between the points $f(x)$ and $f(z)$.
- The right side, $f(\alpha x + (1-\alpha)z)$, represents the actual function evaluated at that intermediate point.

If the inequality holds strictly, the function is strictly convex. Conversely, a function $f$ is defined as concave if and only if $-f$ is convex. Because of this rigid geometric structure, the only functions that are simultaneously both convex and concave are linear (affine) functions. As it is often said, convex optimization is a "one-sided world": the global maximum of a convex function in $\mathbb{R}^n$ is typically $+\infty$ (unless the function is a constant).

## Convexity and First-Order Derivatives

When a convex function is differentiable ($f \in C^1$), we can analyze its convexity through its gradient $\nabla f$. 

The first crucial property is that the gradient of a convex function is **monotone**. This means the direction of the gradient consistently points away from the minimum:
$$\langle \nabla f(z) - \nabla f(x), z - x \rangle \ge 0 \quad \forall x, z$$

More importantly, the **first-order model** (the tangent hyperplane) of a convex function always acts as a global lower bound for the entire function:
$$L_x(z) = f(x) + \langle \nabla f(x), z-x \rangle \le f(z)$$
- Geometrically, this means the epigraph of the function (the set of points above the graph, $epi(f)$) is contained within a half-space defined by the first-order model ($epi(L_x) \supseteq epi(f)$).
- This property leads to the most important theorem in convex optimization: if we find a stationary point where the gradient is zero ($\nabla f(x) = 0$), the inequality collapses to $f(x) \le f(z) \forall z \in \mathbb{R}^n$. 
- Therefore, **every stationary point of a convex function is guaranteed to be a global minimum**.

## Convexity and Second-Order Derivatives

If the function is twice differentiable ($f \in C^2$), checking for convexity becomes an eigenvalue problem using the Hessian matrix $\nabla^2 f(x)$.

A $C^2$ function is convex if and only if its Hessian is positive semi-definite everywhere in its domain:
$$\nabla^2 f(x) \ge 0 \quad \forall x \in \mathbb{R}^n$$
- The absolute best-case scenario for optimization algorithms is when the Hessian is strictly positive definite, specifically bounded away from zero: $\nabla^2 f \ge \tau I$ with $\tau > 0$. This indicates strong convexity.
- Computing the Hessian and proving it is positive semi-definite is often the standard mathematical way to prove a function is convex, unless the function is built by combining known convex functions.

## Basic Convex Functions
There is a foundational library of functions that are intrinsically convex. Recognizing these is crucial:
- **Affine functions:** $f(x) = bx + c$ (these are both convex and concave).
- **Quadratic functions:** $f(x) = \frac{1}{2}x^T Q x + qx$, but only if the matrix $Q$ is positive semi-definite ($Q \ge 0$).
- **Exponential:** $f(x) = e^{ax}$ for any real scalar $a \in \mathbb{R}$.
- **Negative Logarithm:** $f(x) = -\ln(x)$ (restricted to the domain $x > 0$).
- **Power functions:** $f(x) = x^a$ (restricted to $x \ge 0$) only if $a \ge 1$ or $a \le 0$.
- **Norms:** Any $p$-norm $f(x) = ||x||_p$ is convex for $p \ge 1$.
- **Max function:** $f(x) = \max\{x_1, ..., x_n\}$.
- **Matrix eigenvalues:** For a symmetric matrix $Q \in \mathbb{R}^{n \times n}$ with ordered eigenvalues $\lambda_1 \ge \lambda_2 \ge ... \ge \lambda_n$, the sum of the $m$ largest eigenvalues is convex: $f_m(Q) = \sum_{i=1}^m \lambda_i$.

## Convexity-Preserving Operations
Instead of computing complex Hessians, we can guarantee convexity if a complex function is built using operations that preserve convexity:
- **Non-negative combination:** If $f$ and $g$ are convex, and $\delta, \beta \in \mathbb{R}_+$, then $\delta f + \beta g$ is convex.
- **Supremum:** The maximum over an arbitrary (even infinite) set of convex functions is convex: $f(x) = \sup_{i \in I} \{f_i(x)\}$.
- **Pre-composition with linear mapping:** If $f$ is convex, applying a linear transformation to the input keeps it convex: $f(Ax+b)$.
- **Post-composition:** If $f$ is convex and $g$ is a convex *and increasing* function, then $g(f(x))$ is convex.
- **Infimal convolution:** $f(x) = \inf\{f_1(x_1) + f_2(x_2) : x_1 + x_2 = x\}$ is convex if $f_1$ and $f_2$ are convex.
- **Partial minimization:** The value function of a convex constrained problem is convex: $f(x) = \inf\{g(x,z) : z \in \mathbb{R}^m\}$ is convex if $g(x,z)$ is convex.
- **Perspective (Dilation):** If $f(x)$ is convex, its perspective function $p(x,u) = u f(x/u)$ is convex on the domain $u > 0$.

## Why Convex and Not Unimodal or Quasiconvex?
In standard 1D calculus ($n=1$), we often settle for "unimodal" functions (functions with a single valley). In multivariate spaces, the closest equivalent is a **quasiconvex** function, defined as:
$$\alpha f(x) + (1-\alpha)f(z) \le \max\{f(x), f(z)\}$$

A function is quasiconvex if and only if every nonempty sublevel set $S(f,l) = \{x : f(x) \le l\}$ is a convex set (or a possibly infinite interval). 
- While every convex function is quasiconvex ($f \text{ convex} \Rightarrow f \text{ quasiconvex}$), the reverse is strictly not true.
- The major issue with quasiconvex functions is that their algebraic properties are much "weaker". For example, while multiplying a quasiconvex function by a positive scalar $\delta$ yields another quasiconvex function, **the sum of two quasiconvex functions is generally false** (it is not guaranteed to be quasiconvex). 
- Because their algebra breaks down under addition, there is no reliable framework for "Disciplined QuasiConvex Programming", making them significantly harder to optimize in complex machine learning models compared to strictly convex functions.
# References