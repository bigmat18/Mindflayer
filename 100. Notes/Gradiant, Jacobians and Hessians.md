---
Data:
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Unconstrained Multivariate Optimality and Convexity]]"
Area: "[[Master's degree]]"
---
# Unconstrained Multivariate Optimization: Global vs. Local

When moving from one-dimensional to multivariate optimization, our objective function becomes $f:\mathbb{R}^{n}\rightarrow\mathbb{R}$, which can be written as $f(x_{1},x_{2},...,x_{n})=f(x)$. While the core goal remains finding the minimum value, the "geometry" of $\mathbb{R}^n$ introduces significant theoretical and computational hurdles.

---

## 1. Unconstrained Global Optimization

Finding the global minimum—the absolute lowest point—in multiple dimensions is an incredibly daunting task. To even stand a chance, we must assume that $f$ is at least $L$-continuous (Lipschitz continuous), meaning the function doesn't change too abruptly within a bounded interval.

### The Problem: The Curse of Dimensionality
There is a fundamental theoretical limit here: no algorithm can work in less than $\Omega((LD/\epsilon)^{n})$.
This means the computational effort required grows **exponentially** with the number of variables $n$.
- **Curse of dimensionality:** Global optimization isn't really doable unless $n$ is very small, typically $n=3, 5$, or $10$ at most.
- For high precision (small $\epsilon$) or large domains ($D$), the number of points to check becomes astronomical.

### Global Search Strategies
Despite these difficulties, several practical approaches exist to hunt for the global minimum:
- **Multidimensional Grid Search:** You can find a solution in $O((LD/\epsilon)^{n})$ using a multidimensional grid with a small enough step. This is the standard approach to hyperparameter optimization, though the constants $D$ (domain diameter) and $L$ are often unknown.
- **Analytic Functions:** If the analytical form of $f$ is known, clever spatial Branch & Bound (B&B) algorithms can isolate the global optimum.
- **Black-box (Heuristics):** If the function is a "black box" (typically with no derivatives available), many effective heuristics can give good, though not provably optimal, solutions.

> **Key Takeaway:** Finding good global solutions is hard in practice, and proving optimality is even worse unless $f$ is convex. If the function is convex, we have the mathematical guarantee that **global = local**, which makes everything drastically simpler.

---

## 2. Unconstrained Local Optimization

Since the global problem is often intractable, we often settle for local optimization—finding a minimum within a specific "neighborhood." Here, the situation improves significantly.

### Dimension Independence
Unlike the global case, local optimization is much more efficient.
- Results are generally surprisingly analogous to the (multivariate) quadratic case.
- **Dimension-independent complexity:** Most (but not all) convergence results do not explicitly depend on $n$ (or if they do, not exponentially).
- This happens because almost all local algorithms are built on **linear or quadratic models**, which are staples of the field.

### Computational Reality and Limits
Just because the theory says convergence is "dimension-independent" doesn't mean the algorithm is instantaneous:
- **Convergence speed:** It can still be quite low ("badly linear" or worse).
- **Iteration cost:** The cost of computing $f(x)$ and its derivatives necessarily increases with $n$. For large scale problems ($n\approx10^{9}$), even $O(n^{2})$ is too much.
- Some dependency on $n$ might be hidden within the $O(\cdot)$ constants.

Despite this, **large-scale local optimization is doable if you have derivatives**. However, derivatives in $\mathbb{R}^{n}$ are significantly more complex than in the one-dimensional case.

---

# Gradients, Jacobians, and Hessians: Tools for Multivariate Optimization

To optimize a function in multiple dimensions, we must understand its local behavior. This requires extending the concepts of limits, continuity, and derivatives from $\mathbb{R}$ to $\mathbb{R}^n$.

---

## 3. Mathematical Topology and Limits in $\mathbb{R}^n$

Before calculating derivatives, we need a rigorous way to define what it means for points to be "close" to one another in $\mathbb{R}^n$.

- **The Ball:** The fundamental concept is the ball, defined by a center $x \in \mathbb{R}^n$ and a radius $r > 0$: 
  $\mathcal{B}(x,r):=\{z\in\mathbb{R}^{n}:||z-x||\le r\}$

The notion of distance $||\cdot||$ depends on the specific norm being used. The Euclidean norm is just one member of a large family known as $p$-norms:
- **$p$-norm ($p > 0$):** $||x||_{p}:=(\sum_{i=1}^{n}|x_{i}|^{p})^{1/p}$
- **Euclidean norm:** $\equiv||x||_{2}$
- **Lasso norm (Manhattan):** $||x||_{1}:=\sum_{i=1}^{n}|x_{i}|$
- **Infinity norm (Max norm):** $\lim_{p\rightarrow\infty}\equiv||x||_{\infty}:=\max\{|x_{i}|:i=1,...,n\}$
- **Zero "norm" (counts non-zeros):** $\lim_{p\rightarrow0}\equiv||x||_{0}:=\#\{i:|x_{i}|>0\}$

The norm defines the topology of $\mathbb{R}^n$, but in practice, it doesn't really matter which one you choose because all norms are equivalent:
$\forall||\cdot||,|||\cdot|||\exists0<\alpha<\beta~s.t.\alpha||x||\le|||x|||\le\beta||x||\forall x,z\in\mathbb{R}^{n}$

### Limits and Continuity
The limit of a sequence $\{x_i\} \subset \mathbb{R}^n$ is written as:
$\lim_{i\rightarrow\infty}x_{i}=x\equiv\{x_{i}\}\rightarrow x$

This means that eventually, all points in the sequence come arbitrarily close to $x$:
$\forall\epsilon>0\exists h\text{ s.t. }x_{i}\in\mathcal{B}(x,\epsilon)\forall i\ge h$, which is equivalent to saying $\lim_{i\rightarrow\infty}d(x_{i},x)=0$.

A function $f$ is **continuous** at $x$ if:
$\{x_{i}\}\rightarrow x\Rightarrow\{f(x_{i})\}\rightarrow f(x)$
If it is continuous everywhere, we write $f\in C^{0}$.

**The Dimensionality Trap:** Space in $\mathbb{R}^n$ is "exponentially larger" than in $\mathbb{R}$, meaning there are many more ways for $\{x_i\} \to x$. The limit must be the same for *all* possible paths. 
Consider the tricky function $f(x_{1},x_{2})=[\frac{x_{1}^{2}x_{2}}{x_{1}^{4}+x_{2}^{2}}]^{2}$:
- If we approach $(0,0)$ on straight lines ($\forall[d_{1},d_{2}]\in\mathbb{R}^{2}$), the limit is $0$: $\lim_{k\rightarrow\infty}f(d_{1}/k,d_{2}/k)=0$.
- However, if we approach on a curved line, the limit changes: $\lim_{k\rightarrow\infty}f(1/k,1/k^{2})=1/4$.
This shows why non-differentiability in $\mathbb{R}^n$ can lead to tricky situations.

---

## 4. Directional Derivatives, Partial Derivatives, and the Gradient

How does our function $f$ change if we move away from $x$ along a specific direction $d \in \mathbb{R}^n$? 
- **Directional derivative:** Evaluated at $x \in \mathbb{R}^n$ along direction $d \in \mathbb{R}^n$:
  $\frac{\partial f}{\partial d}(x):=\lim_{t\rightarrow0}\frac{f(x+td)-f(x)}{t}=\varphi_{x,d}^{\prime}(0)$
  This scales linearly with the magnitude of $d$: $\frac{\partial f}{\partial\beta d}(x)=\beta\frac{\partial f}{\partial d}(x)$.

- **Partial derivative:** A special case where the direction is aligned with one of the coordinate axes ($X_j$). 
  $\frac{\partial f}{\partial x_{i}}(x):=\lim_{t\rightarrow0}\frac{f(x_{1},...,x_{i-1},x_{i}+t,x_{i+1},...,x_{n})-f(x)}{t}=[f_{x}^{i}]^{\prime}(x_{i})$
  This is easy to compute: just treat all $x_j$ for $j \neq i$ as constants.

- **Gradient:** The gradient is simply the column vector grouping all these partial derivatives together. It represents the generalized first derivative:
  $\nabla f(x):=[\frac{\partial f}{\partial x_{1}}(x),...,\frac{\partial f}{\partial x_{n}}(x)]^{T}\in\mathbb{R}^{n}$

Important foundational examples:
- Linear function: $f(x)=\langle b,x\rangle\Rightarrow\nabla f(x)=b$
- Quadratic function: $f(x)=\frac{1}{2}x^{T}Qx+qx\Rightarrow\nabla f(x)=Qx+q$

---

## 5. Differentiability and the First-Order Model

Merely having partial derivatives does not guarantee that a function is smooth in $\mathbb{R}^n$. A function $f$ is truly **differentiable** at $x$ if there exists a linear function $\phi(h)=\langle b,h\rangle+f(x)$ such that the error vanishes "faster than linearly":
$\lim_{||h||\rightarrow0}\frac{|f(x+h)-\phi(h)|}{||h||}=0$

When $f$ is differentiable at $x$:
- The vector $b$ perfectly matches the gradient: $b=\nabla f(x)$.
- We can construct the **first-order model** of $f$ at $x$:
  $L_{x}(z)=\langle\nabla f(x),z-x\rangle+f(x)$
- The gradient gives us *all* directional derivatives:
  $\forall d\in\mathbb{R}^{n}$ $\frac{\partial f}{\partial d}(x)=\langle\nabla f(x),d\rangle$
- It guarantees continuity: $f$ differentiable at $x \Rightarrow f$ continuous at $x$.
- If the partial derivatives $\frac{\partial f}{\partial x_{i}}\in C^{0}$, then $f$ is differentiable everywhere ($\equiv f\in C^{1}$).

**Geometric Interpretation:**
In $\mathbb{R}^n$, the level set $L(L_x, f(x))$ is a surface passing by $x$, and the gradient is orthogonal to it. If $f$ is differentiable at $x$: 
$L(L_{x},f(x))\perp L(f,f(x))\perp\nabla f(x)$
If $f$ is non-differentiable at $x$ (like $f(x_{1},x_{2})=||[x_{1},x_{2}]||_{1}=|x_{1}|+|x_{2}|$), the level surface has "kinks", and things break down.

---

## 6. Derivatives of Vector-Valued Functions: The Jacobian

When dealing with vector-valued functions $f: \mathbb{R}^n \to \mathbb{R}^m$, where $f(x) = [f_1(x), f_2(x), \dots, f_m(x)]$, the partial derivative handles the extra index seamlessly:
$\frac{\partial f_{j}}{\partial x_{i}}(x)=\lim_{t\rightarrow0}\frac{f_{j}(x_{1},...,x_{i-1},x_{i}+t,x_{i+1},...,x_{n})-f_{j}(x)}{t}$

Grouping all $m \times n$ partial derivatives gives us the **Jacobian** matrix. It is an $m \times n$ matrix with the gradients of each scalar component $f_j$ acting as its rows:

$$Jf(x) := \begin{bmatrix}\nabla f_1(x)^{T}\\ \nabla f_2(x)^{T}\\ \vdots\\ \nabla f_m(x)^{T}\end{bmatrix}$$

---

## 7. Second-Order Derivatives, Hessians, and the Second-Order Model

Because $\frac{\partial f}{\partial x_{i}}:\mathbb{R}^{n}\rightarrow\mathbb{R}$ is itself a function, it has partial derivatives of its own. If we differentiate twice, we obtain the **second order partial derivative**:
$\frac{\partial^{2}f}{\partial x_{j}\partial x_{i}}$ and $\frac{\partial^{2}f}{\partial x_{i}\partial x_{i}}=\frac{\partial^{2}f}{\partial x_{i}^{2}}=[f_{x}^{i}]^{\prime\prime}$

By taking the Jacobian of the gradient map $\nabla f(x):\mathbb{R}^{n}\rightarrow\mathbb{R}^{n}$, we compute the **Hessian (matrix)** of $f$ at $x$:

$$\nabla^2 f(x) := \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2}(x) & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_n}(x) \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1}(x) & \dots & \frac{\partial^2 f}{\partial x_n^2}(x) \end{bmatrix}$$

For a quadratic function $f(x)=\frac{1}{2}x^{T}Qx+qx$, the Hessian elegantly simplifies: $\nabla^{2}f(x)=Q$.

Using the Hessian, we can build the **second-order model**, which acts as a much better local approximation:
$Q_{x}(z)=L_{x}(z)+\frac{1}{2}(z-x)^{T}\nabla^{2}f(x)(z-x)$

### Computational Cost, Symmetry, and $C^2$ Functions
Calculating the Hessian requires $O(n^2)$ memory to store and compute (unless sparse), making it bad when $n$ is large.

- **Symmetry:** If $\exists\delta>0$ such that $\forall z\in\mathcal{B}(x,\delta)$ the mixed partials exist and are continuous at $x$, then the Hessian is symmetric:
  $\frac{\partial^{2}f}{\partial x_{j}\partial x_{i}}(x)=\frac{\partial^{2}f}{\partial x_{i}\partial x_{j}}(x)\equiv\nabla^{2}f \text{ symmetric}$
- A symmetric Hessian guarantees that all eigenvalues of $\nabla^{2}f(x)$ are real.
- **The $C^2$ Class:** $f\in C^{2}:=\nabla^{2}f(x)$ continuous everywhere. This implies $\nabla f(x)\in C^{1}\Rightarrow\nabla f(x)\in C^{0}\Rightarrow f(x)\in C^{0}$. The $C^2$ class is the best class ever for optimization, but it is sometimes necessary to make do with less.
# References