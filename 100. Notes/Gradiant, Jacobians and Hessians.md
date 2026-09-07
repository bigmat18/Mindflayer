---
Data: 2026-03-22T15:33:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Unconstrained Multivariate Optimality and Convexity]]"
Area: "[[Master's degree]]"
---
# Gradients, Jacobians, and Hessians

To optimize a function in multiple dimensions, we must understand its local behavior. This requires extending the concepts of limits, continuity, and derivatives from $\mathbb{R}$ to $\mathbb{R}^n$.

## Mathematical Topology and Limits in $\mathbb{R}^n$

Before calculating derivatives, we need a rigorous way to define what it means for points to be "close" to one another in $\mathbb{R}^n$.

**The Ball:** The fundamental concept is the ball, defined by a center $x \in \mathbb{R}^n$ and a radius $r > 0$: 
$$\mathcal{B}(x,r) := \{ z \in \mathbb{R}^{n} : ||z-x|| \le r \}$$

The notion of distance $|| \cdot ||$ depends on the specific norm being used. The Euclidean norm is just one member of a large family known as $p$-norms.

- **$p$-norm ($p > 0$):** $$||x||_{p} := \left( \sum_{i=1}^{n} |x_{i}|^{p} \right)^{1/p}$$
- **Euclidean norm:** $\equiv ||x||_{2}$
- **Lasso norm (Manhattan):** $$||x||_{1} := \sum_{i=1}^{n} |x_{i}|$$
- **Infinity norm (Max norm):** $$\lim_{p\rightarrow\infty} \equiv ||x||_{\infty} := \max\{|x_{i}| : i=1,...,n\}$$
- **Zero "norm" (counts non-zeros):** $$\lim_{p\rightarrow0} \equiv ||x||_{0} := \#\{i : |x_{i}| > 0\}$$

The norm defines the topology of $\mathbb{R}^n$, but in practice, it doesn't really matter which one you choose because all norms are mathematically equivalent in finite dimensions:
$$\forall ||\cdot||, |||\cdot||| \exists 0 < \alpha < \beta \text{ s.t. } \alpha||x|| \le |||x||| \le \beta||x|| \quad \forall x, z \in \mathbb{R}^{n}$$

### Limits and Continuity
The limit of a sequence $\{x_i\} \subset \mathbb{R}^n$ is written as:
$$\lim_{i\rightarrow\infty} x_{i} = x \equiv \{x_{i}\} \rightarrow x$$

This means that eventually, all points in the sequence come arbitrarily close to $x$. This can be formally written in equivalent ways:
$$\Longleftrightarrow \forall \epsilon > 0 \exists h \text{ s.t. } d(x_i, x) \le \epsilon \quad \forall i \ge h$$
$$\Longleftrightarrow \forall \epsilon > 0 \exists h \text{ s.t. } x_i \in \mathcal{B}(x, \epsilon) \quad \forall i \ge h$$
$$\Longleftrightarrow \lim_{i\to\infty} d(x_i, x) = 0$$

A function $f$ is continuous at $x$ if:
$$\{x_{i}\} \rightarrow x \Rightarrow \{f(x_{i})\} \rightarrow f(x)$$
If it is continuous everywhere, we write $f \in C^{0}$. This notations means however the function converge to x

**The Dimensionality Trap:** Space in $\mathbb{R}^n$ is "exponentially larger" than in $\mathbb{R}$, meaning there are infinitely many more ways or paths for $\{x_i\} \to x$. The limit must be exactly the same for *all* possible paths.

Consider the tricky function:
$$f(x_{1},x_{2}) = \left[ \frac{x_{1}^{2}x_{2}}{x_{1}^{4}+x_{2}^{2}} \right]^{2}$$
- If we approach $(0,0)$ on straight lines ($\forall[d_{1},d_{2}]\in\mathbb{R}^{2}$), the limit is $0$: $$\lim_{k\rightarrow\infty} f(d_{1}/k, d_{2}/k) = 0$$
- However, if we approach on a curved line (e.g., $x_2 = x_1^2$), the limit changes: $$\lim_{k\rightarrow\infty} f(1/k, 1/k^{2}) = 1/4$$
This shows why non-differentiability in $\mathbb{R}^n$ can lead to tricky, counterintuitive situations.
![[Pasted image 20260322155141.png | 400]]

## [[Gradient]]

## [[Jacobian]]

## [[Hessians]]

# References