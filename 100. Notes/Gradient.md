---
Data: 2026-03-22T17:27:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Gradient
How does our function $f$ change if we move away from $x$ along a specific direction $d \in \mathbb{R}^n$? 
- **Directional derivative:** Evaluated at $x \in \mathbb{R}^n$ along direction $d \in \mathbb{R}^n$:$$\frac{\partial f}{\partial d}(x) := \lim_{t\rightarrow0} \frac{f(x+td)-f(x)}{t} = \varphi_{x,d}^{\prime}(0)$$
  This scales linearly with the magnitude of $d$: $$\frac{\partial f}{\partial\beta d}(x) = \beta\frac{\partial f}{\partial d}(x)$$
- **Partial derivative:** A special case where the direction is perfectly aligned with one of the coordinate axes ($x_i$).$$\frac{\partial f}{\partial x_{i}}(x) := \lim_{t\rightarrow0} \frac{f(x_{1},...,x_{i-1},x_{i}+t,x_{i+1},...,x_{n})-f(x)}{t} = [f_{x}^{i}]^{\prime}(x_{i})$$
  This is easy to compute: just treat all $x_j$ (for $j \neq i$) as constants.

- **Gradient:** The gradient is simply the column vector grouping all these partial derivatives together. It represents the generalized first derivative in $\mathbb{R}^n$:$$\nabla f(x) := \left[ \frac{\partial f}{\partial x_{1}}(x), ..., \frac{\partial f}{\partial x_{n}}(x) \right]^{T} \in \mathbb{R}^{n}$$
Important foundational examples:
- Linear function: $f(x) = \langle b,x \rangle \Rightarrow \nabla f(x) = b$
- Quadratic function: $f(x) = \frac{1}{2}x^{T}Qx + qx \Rightarrow \nabla f(x) = Qx + q$

#### Differentiability in $\mathbb{R}^n$
Merely having partial derivatives does not guarantee that a function is completely smooth in $\mathbb{R}^n$. A function $f$ is truly **differentiable** at $x$ if there exists a linear function $\phi(h) = \langle b,h \rangle + f(x)$ such that the approximation error vanishes "faster than linearly":
$$\lim_{||h||\rightarrow0} \frac{|f(x+h)-\phi(h)|}{||h||} = 0 \:\:\:[\Longrightarrow \phi(0) = f(0) \Longrightarrow c=f(x)]$$
$\varphi \equiv$ "first order moel" of $f$ at $x$, the **error** "vanishes faster than linearly"  

When $f$ is differentiable at $x$:
- The vector $b$ perfectly matches the gradient: $b = \nabla f(x)$.
- We can construct the **first-order model** (tangent hyperplane) of $f$ at $x$:  $$L_{x}(z) = \langle \nabla f(x), z-x \rangle + f(x)$$
- The gradient gives us *all* directional derivatives simultaneously: $$\forall d \in \mathbb{R}^{n} \quad \frac{\partial f}{\partial d}(x) = \langle \nabla f(x), d \rangle$$
- It guarantees continuity: $f$ differentiable at $x \Rightarrow f$ continuous at $x$.
- If the partial derivatives are continuous ($\frac{\partial f}{\partial x_{i}} \in C^{0}$), then $f$ is differentiable everywhere ($\equiv f \in C^{1}$).
###### Non-differentiability Example 1
First let's consider the function:
$$
f(x_1, x_2) = ||[x_1. x_2]||_1 = |x_1| + |x_2|
$$
$f$ is continuous everywhere but why? We can see the gradiants:
$$
\exists d \in \mathbb{R}^n \:\: s.t. \nexists \frac{\partial f}{\partial d}(0,0)
$$
we can see that f is non differentiable in $[0,0]$

![[Pasted image 20260322163921.png | 350]]

###### Non-differentiability Example 2
Now let's consider the function:
$$f(x_1, x_2) = \frac{x_1^2 x_2}{x_1^2 + x_2^2}$$
can we take $f(0,0)=0$ as:
$$
\lim_{[x_1, x_2] \to [0,0]} f(x_1, x_2)=0
$$
in this case $\exists \frac{\partial f}{\partial d} \forall d \in \mathbb{R}^n$ but $f$ non differetiable in $[0,0]$

![[Pasted image 20260322164315.png | 350]]

###### Non-differentiability Example 3
Now let's consider the function:
$$
f(x_1, x_2) = \bigg[ \frac{x_1^2 x_2}{x_1^4 + x_2^2} \bigg]^2
$$
if **f non continuos** than it is not differentiable at $[0,0]$. In this case we have:
$$
\frac{\partial f}{\partial d}(0,0) = 0 \:\: \forall d \in \mathbb{R}^n
$$
$\nexists \nabla f$  but $\exists v = 0$ s.t. $\frac{\partial f}{\partial d} = <v, d> \forall d\in \mathbb{R}^n$. f does nasty things on **curved lines** not stright ones.

![[Pasted image 20260322170209.png |350]]

#### The Gradient in $\mathbb{R}^n$
To fully grasp optimization, it is essential to geometrically visualize what the gradient represents relative to the function's surface.

In $\mathbb{R}^n$, the level set $L(L_x, f(x))$ **represents a hyperplane (or a surface)** passing through the point $x$. The fundamental geometric property is that the gradient $\nabla f(x)$ is always strictly orthogonal to this surface: 
$$\nabla f(x) \perp L(L_{x},f(x))$$
Let's consider this function:
$$
f(x_1, x_2) = \frac{x_1^2 x_2}{x_1^2 + x_2^2}
$$
Its gradient can be computed using standard derivative rules:
$$\nabla f(x) = \left[ \frac{2x_1 x_2^3}{(x_1^2 + x_2^2)^2}, \frac{x_1^2(x_1^2 - x_2^2)}{(x_1^2 + x_2^2)^2} \right]^T$$
![[Pasted image 20260322170347.png | 300]]

If $f$ is **differentiable** at $x$, a perfect geometric harmony emerges. The level set of the linear model, the level set of the function itself, and the gradient are linked by orthogonality:
$$L(L_{x},f(x)) \perp L(f,f(x)) \perp \nabla f(x)$$
Under these conditions, the level curve $L(f,f(x))$ is perfectly **"smooth"**.

![[Pasted image 20260322170431.png | 300]]

If a function is non-differentiable at a point this neat geometric relationship completely breaks down. As $x \to \overline{x}$ where $f$ is non-differentiable, the level surface $L(f, f(x))$ becomes **"less and less smooth"**.

Level set close to 0:
![[Pasted image 20260322170543.png | 300]]

Exactly at the point of **non-differentiability**, the level surface develops **"kinks"** (sharp corners). When these kinks appear, it is impossible to define a single tangent plane or a unique gradient vector. 

Level set at 0:
![[Pasted image 20260322170600.png | 300]]

All relevant objects in $\mathbb{R}^{n+1}$ and $\mathbb{R}^n$ lose their smoothness, and the standard calculus machinery breaks. $f$ non differentialbe $\Longrightarrow$ **kinks** appear and things break


# References