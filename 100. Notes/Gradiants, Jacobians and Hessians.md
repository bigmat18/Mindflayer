---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# Gradiants, Jacobians and Hessians

## Mathematically Speaking: Hints of Topology in $\mathbb{R}^n$

To understand derivatives in higher dimensions, we must first establish the topology of $\mathbb{R}^n$. The fundamental, yet easy, concept is the definition of a "ball", which groups all points $z$ that are "close" to a center $x \in \mathbb{R}^n$ within a radius $r > 0$ according to a chosen norm:
$$\mathcal{B}(x,r):=\{z\in\mathbb{R}^{n}:||z-x||\le r\}$$

The Euclidean norm is just one member of a large family of norms known as $p$-norms, defined for $p > 0$:
$$||x||_{p}:=\left(\sum_{i=1}^{n}|x_{i}|^{p}\right)^{1/p}$$
- The standard Euclidean norm is equivalent to the 2-norm: Euclidean $\equiv||x||_{2}$.
- The $L_1$ norm, often used in Lasso regression, is defined as: $||x||_{1}:=\sum_{i=1}^{n}|x_{i}|$.
- The maximum norm is defined via a limit: $\lim_{p\rightarrow\infty}\equiv||x||_{\infty}:=\max\{|x_{i}|:i=1,...,n\}$.
- The $L_0$ formulation, which simply counts non-zero elements (and is technically not a true norm), is defined as: $\lim_{p\rightarrow0}\equiv||x||_{0}:=\#\{i:|x_{i}|>0\}$.
- These norms can be visualized as growing sets $S(||\cdot||_{p},1)\equiv\mathcal{B}_{p}(0,1)$ for $p=0,1/2,1,3/2,2,3,\infty$, which grow with $p$.

While the choice of norm changes the specific shape of the unit ball, it defines the topology of $\mathbb{R}^n$ but fundamentally doesn't really matter. This is because all is "ball" or "small ball", and all norms are mathematically equivalent:
$$\forall||\cdot||,|||\cdot|||\exists0<\alpha<\beta \text{ s.t. } \alpha||x||\le|||x|||\le\beta||x||\forall x,z\in\mathbb{R}^{n}$$

## Limit of a Sequence and Continuity in $\mathbb{R}^n$

Using this topology, we define the limit of a sequence $\{x_{i}\}\subset\mathbb{R}^{n}$ as:
$$\lim_{i\rightarrow\infty}x_{i}=x\equiv\{x_{i}\}\rightarrow x$$
This means that points of $\{x_i\}$ eventually all come arbitrarily close to $x$, which can be formally expressed in three equivalent ways:
- $\forall\epsilon>0\exists h \text{ s.t. } d(x_{i},x)\le\epsilon\forall i\ge h$
- $\forall\epsilon>0\exists h \text{ s.t. } x_{i}\in\mathcal{B}(x,\epsilon)\forall i\ge h$
- $\lim_{i\rightarrow\infty}d(x_{i},x)=0$

Because $\mathbb{R}^n$ is "exponentially larger" than $\mathbb{R}$, there are many more ways for $\{x_{i}\}\rightarrow x$ to occur, leading to more tricky situations and concepts. 

This directly impacts the definition of continuity. A function $f$ is continuous at $x$ if:
$$\{x_{i}\}\rightarrow x\Rightarrow\{f(x_{i})\}\rightarrow f(x)$$
If this holds everywhere, $f \in C^0$ (continuous $\forall x\in\mathbb{R}^{n}$). Because there are "many" different $\{x_i\} \to x$, the limit must be strictly equal for all of them; it is not sufficient to only consider "simple" sequences. 

To demonstrate this, consider evaluating $f(0,0)$ for the function:
$$f(x_{1},x_{2})=\left[\frac{x_{1}^{2}x_{2}}{x_{1}^{4}+x_{2}^{2}}\right]^{2}$$
- If we approach the origin "on straight lines" using any generic direction $\forall[d_{1},d_{2}]\in\mathbb{R}^{2}$, the result is zero: $\lim_{k\rightarrow\infty}f(d_{1}/k,d_{2}/k)=0$.
- However, if we approach on a "curved" line, it yields a different result: $\lim_{k\rightarrow\infty}f(1/k,1/k^{2})=1/4$.

Because the limit $\neq$ on curved lines compared to straight lines, the function is discontinuous at $(0,0)$.

## Directional and Partial Derivatives

The directional derivative of a function $f:\mathbb{R}^{n}\rightarrow\mathbb{R}$ at a point $x\in\mathbb{R}^{n}$ along a direction $d\in\mathbb{R}^{n}$ is the derivative of the $(x, d)$-tomography evaluated at $0$:
$$\frac{\partial f}{\partial d}(x):=\lim_{t\rightarrow0}\frac{f(x+td)-f(x)}{t}=\varphi_{x,d}^{\prime}(0)$$
- This metric scales linearly with the magnitude of $d$: $\frac{\partial f}{\partial\beta d}(x)=\beta\frac{\partial f}{\partial d}(x)$.
- The one-sided directional derivative is defined similarly as: $\lim_{t\rightarrow0_{\pm}}...=[\varphi_{x,d}]_{\pm}^{\prime}(0)$.

A partial derivative is a special case of the directional derivative evaluated specifically with respect to one variable $x_i$, treating all other variables $X_j$ (for $j\ne i$) as constants:
$$\frac{\partial f}{\partial x_{i}}(x):=\lim_{t\rightarrow0}\frac{f(x_{1},...,x_{i-1},x_{i}+t,x_{i+1},...,x_{n})-f(x)}{t}=[f_{x}^{i}]^{\prime}(x_{i})=\frac{\partial f}{\partial u^{i}}(x)$$

The **Gradient** is formed by collecting all these partial derivatives into a (column) vector:
$$\nabla f(x):=\left[\frac{\partial f}{\partial x_{1}}(x),...,\frac{\partial f}{\partial x_{n}}(x)\right]^{T}\in\mathbb{R}^{n}$$
- For a linear function $f(x)=\langle b,x\rangle\Rightarrow\nabla f(x)=b$.
- For a quadratic function $f(x)=\frac{1}{2}x^{T}Qx+qx\Rightarrow\nabla f(x)=Qx+q$.

## Differentiability in $\mathbb{R}^n$

Differentiability in $\mathbb{R}^n$ requires more than just the existence of partial derivatives. A function $f$ is differentiable at $x$ if there exists a linear function $\phi(h)=\langle b,h\rangle+f(x)$ such that the error between the function and this model "vanishes faster than linearly":
$$\lim_{||h||\rightarrow0}\frac{|f(x+h)-\phi(h)|}{||h||}=0 [\Rightarrow\phi(0)=f(x)\Rightarrow c=f(x)]$$

If $f$ is differentiable at $x$, it implies several critical properties:
- The vector $b$ is precisely the gradient: $b=\nabla f(x)$.
- We can construct a reliable first-order model of $f$ at $x$: $L_{x}(z)=\langle\nabla f(x),z-x\rangle+f(x)$.
- The gradient $\nabla f(x)$ is sufficient to compute all directional derivatives via inner product: $\forall d\in\mathbb{R}^{n} \frac{\partial f}{\partial d}(x)=\langle\nabla f(x),d\rangle (\leftarrow\exists)$.
- Differentiability guarantees continuity at $x$: $f$ differentiable at $x\Rightarrow f$ continuous at $x$.
- If partial derivatives are continuous ($\frac{\partial f}{\partial x_{i}}\in C^{0}$), then $f$ is differentiable everywhere ($\equiv f\in C^{1}$).

Non-differentiability in $\mathbb{R}^n$ is much weirder than in $\mathbb{R}$. Consider these three distinct cases where differentiability breaks down:
- **Non-differentiability I:** $f(x_{1},x_{2})=||[x_{1},x_{2}]||_{1}=|x_{1}|+|x_{2}|$. This function is continuous everywhere but is non-differentiable in $[0,0]$.
- **Non-differentiability II:** $f(x_{1},x_{2})=\frac{x_{1}^{2}x_{2}}{x_{1}^{2}+x_{2}^{2}}$. We can take $f(0,0)=0$ as $\lim_{[x_{1},x_{2}]\rightarrow[0,0]}f(x_{1},x_{2})=0$. Even though directional derivatives exist for all directions ($\exists\frac{\partial f}{\partial d}\forall d\in\mathbb{R}^{n}$), $f$ remains non-differentiable in $[0,0]$.
- **Non-differentiability III:** $f(x_{1},x_{2})=\left[\frac{x_{1}^{2}x_{2}}{x_{1}^{4}+x_{2}^{2}}\right]^{2}$. This function is neither continuous nor differentiable at $[0,0]$. Curiously, $\frac{\partial f}{\partial d}(0,0)=0\forall d\in\mathbb{R}^{n}$, but there is no vector $v$ such that $\frac{\partial f}{\partial d}=\langle v,d\rangle\forall d\in\mathbb{R}^{n}$.

Geometrically, in $\mathbb{R}^n$, $L(L_{x},f(x))$ represents a surface passing by $x$, and the gradient is orthogonal to it: $\nabla f(x)\perp L(L_{x},f(x))$. If $f$ is differentiable at $x$, then $L(L_{x},f(x))\perp L(f,f(x))\perp\nabla f(x)$ and $L(f,f(x))$ is "smooth". Conversely, as $x\rightarrow\overline{x}$ where $f$ is non-differentiable, $L(f,f(x))$ becomes "less and less smooth", developing "kinks" where things break.

## Derivatives of Vector-Valued Functions and the Jacobian

When analyzing vector-valued functions $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$, the output is a vector: $f(x) = [f_1(x), f_2(x), ..., f_m(x)]$. The partial derivative involves an extra index to track the specific component function:
$$\frac{\partial f_{j}}{\partial x_{i}}(x)=\lim_{t\rightarrow0}\frac{f_{j}(x_{1},...,x_{i-1},x_{i}+t,x_{i+1},...,x_{n})-f_{j}(x)}{t}$$

The **Jacobian** is the $m \times n$ matrix containing all $m\cdot n$ partial derivatives, structured with gradients as its rows:
$$Jf(x) := \begin{bmatrix} \nabla f_1(x)^T \\ \nabla f_2(x)^T \\ \vdots \\ \nabla f_m(x)^T \end{bmatrix}$$
This matrix framework will come in handy later on for constrained optimization.

## Second-Order Derivatives and the Hessian

Because the partial derivative $\frac{\partial f}{\partial x_{i}}:\mathbb{R}^{n}\rightarrow\mathbb{R}$ is itself a function, it has partial derivatives itself. If we "just do it twice", we get the second-order partial derivative:
$$\frac{\partial^{2}f}{\partial x_{j}\partial x_{i}}$$
$$\frac{\partial^{2}f}{\partial x_{i}\partial x_{i}}=\frac{\partial^{2}f}{\partial x_{i}^{2}}=[f_{x}^{i}]^{\prime\prime}$$

Because the gradient is a vector-valued function $\nabla f(x):\mathbb{R}^{n}\rightarrow\mathbb{R}^{n}$, its Jacobian is an $n \times n$ matrix known as the **Hessian (matrix) of $f$ at $x$**:
$$\nabla^2 f(x) := J(\nabla f)(x)$$

- For a linear function $f(x)=\langle b,x\rangle\Rightarrow\nabla^{2}f(x)=0$.
- For a quadratic function $f(x)=\frac{1}{2}x^{T}Qx+qx\Rightarrow\nabla^{2}f(x)=Q$.
- A major computational bottleneck is that the Hessian requires $O(n^{2})$ memory to store and compute (unless it is sparse), which is bad when $n$ is large.

The Hessian is fundamental because it allows us to construct the **second-order model** (which equals the first-order model plus a second-order term), yielding a much better approximation defined by a non-homogeneous quadratic function:
$$Q_{x}(z)=L_{x}(z)+\frac{1}{2}(z-x)^{T}\nabla^{2}f(x)(z-x)$$

Finally, regarding continuity and symmetry, a core theorem states that if $\exists\delta>0$ such that $\forall z\in\mathcal{B}(x,\delta)$ the derivatives $\frac{\partial^{2}f}{\partial x_{j}\partial x_{i}}(z)$ and $\frac{\partial^{2}f}{\partial x_{i}\partial x_{j}}(z)$ exist and are continuous at $x$, then:
$$\frac{\partial^{2}f}{\partial x_{j}\partial x_{i}}(x)=\frac{\partial^{2}f}{\partial x_{i}\partial x_{j}}(x)\equiv\nabla^{2}f \text{ is symmetric}$$
Because it is symmetric, all eigenvalues of $\nabla^{2}f(x)$ are real.

Functions where the Hessian is continuous everywhere belong to the $C^2$ class:
$$f\in C^{2}:=\nabla^{2}f(x) \text{ continuous everywhere } \equiv\partial^{2}f/\partial x_{j}\partial x_{i}\in C^{0}\forall i,j$$
This directly implies that $\nabla^{2}f(x)$ is symmetric everywhere, and also that $\nabla f(x)\in C^{1}\Rightarrow\nabla f(x)\in C^{0}\Rightarrow f(x)\in C^{0}$. Ultimately, $C^2$ (or strictly speaking $C^3$) is the best class ever for optimization.
# References