---
Data: 2026-03-21T13:05:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Unconstrained Multivariate Optimality and Convexity]]"
Area: "[[Master's degree]]"
---
# Optimality Conditions in Multivariate Optimization

Once we know how to compute [[Gradient]] and [[Hessians]], we can use them to find the minimum of a function. Just as in single-variable calculus, we rely on derivatives to pinpoint candidate solutions, but the multivariate landscape introduces new geometrical features like saddle points.

Graphically and intuitively, if a function has a non-zero slope at a point, we can always move downhill. 
- If $f^{\prime}(x)<0$ or $f^{\prime}(x)>0$, $x$ clearly cannot be a local minimum.
![[Pasted image 20260322172853.png | 350]]

- Hence, $f^{\prime}(x)=0$ in all local minima (and consequently, in the global minimum as well).
![[Pasted image 20260322172913.png|350]]

- However, $f^{\prime}(x)=0$ is also true for local (and global) maxima, as well as in saddle points. 
![[Pasted image 20260322172934.png | 400]]

In multiple dimensions, finding a point where the slope is zero in *all* directions gives us a **stationary point**, but we still need to determine whether it is a minimum, a maximum, or a saddle.

## First-Order (Necessary, Local) Optimality Condition
The most fundamental rule of optimization is the first-order necessary condition. If a point is a local minimum, the gradient there must be strictly zero.

#### Theorem
**Theorem:** If $f$ is differentiable at $x$ and $x$ is a local minimum $\Rightarrow \nabla f(x)=0$ (stationary point).

The proof is done by contradiction: suppose $x$ is a local minimum but $\nabla f(x)\ne0$.

Proving $x$ is not a local minimum means showing that for any $\epsilon>0$ "small enough", there exists a point $z\in\mathcal{B}(x,\epsilon)$ such that $f(z)<f(x)$. We have to construct infinitely many $z$ that are better than $x$ and arbitrarily close to it.

Luckily, all these $z$ points can be taken along a single direction $d\in\mathbb{R}^{n}$: $z=x+\alpha d$, with $\alpha>0$.

We can choose the "best" direction $d$, which is the **steepest descent direction** at $x$. This is the normalized anti-gradient:
$$d = -\frac{\nabla f(x)}{||\nabla f(x)||}$$
Along this direction with $||d||=1$, the directional derivative $\frac{\partial f}{\partial d}(x)$ is the most negative.

#### Proof
Let's look at the 1D "tomography" of the function along this negative gradient direction: $\varphi(\alpha)=\varphi_{x,-\nabla f(x)}(\alpha)$. We want to prove that: 
$$\exists\overline{\alpha}>0$ s.t. $\varphi(\alpha)<f(x)=\varphi(0) \quad \forall\alpha\in[0,\overline{\alpha}]$$

We use the definition of $f\in C^{1}$ and its first-order Taylor remainder: $$R(z-x)=f(z)-L_{x}(z)$$By definition, $\lim_{h\rightarrow0} \frac{R(h)}{||h||}=0$, meaning the remainder $R(h)\rightarrow0$ "faster than $h\rightarrow0$".

Expanding $f$ at $z = x - \alpha\nabla f(x)$, we get:
$$\varphi(\alpha) = f(x-\alpha\nabla f(x)) = f(x)+\langle-\alpha\nabla f(x),\nabla f(x)\rangle+R(-\alpha\nabla f(x))$$
$$\varphi(\alpha) = f(x)-\alpha||\nabla f(x)||^{2}+R(-\alpha\nabla f(x))$$

- We have **a negative term** that is linear in $\alpha$ ($-\alpha||\nabla f(x)||^{2}$)
- plus a (possibly) **positive "more than linear"** remainder term ($+R(-\alpha\nabla f(x))$)

As $\alpha\rightarrow0$ (which implies $h = -\alpha\nabla f(x) \to 0$), it is mathematically clear who wins the battle:
$$\lim_{\alpha\rightarrow0}\frac{R(-\alpha\nabla f(x))}{||\alpha\nabla f(x)||}=\lim_{h\rightarrow0}\frac{R(h)}{||h||}=0$$

This means that for any $\epsilon>0$, there exists an $\overline{\alpha}>0$ such that:
$$\frac{R(-\alpha\nabla f(x))}{\alpha||\nabla f(x)||}\le\epsilon \quad \forall\alpha\in[0,\overline{\alpha}]$$

If we specifically choose $\epsilon<||\nabla f(x)||$, we get the inequality $R(-\alpha\nabla f(x))<\alpha||\nabla f(x)||^{2}$.
Substituting this back into our Taylor expansion, we obtain:
$$\varphi(\alpha) = f(x)-\alpha||\nabla f(x)||^{2}+R(-\alpha\nabla f(x)) < f(x)$$

**Conclusion:** A small enough step along $-\nabla f(x)$ (when it is $\ne0$) strictly yields a better point $z$ with a lower function value, directly contradicting the initial assumption that $x$ was a local minimum.


# References