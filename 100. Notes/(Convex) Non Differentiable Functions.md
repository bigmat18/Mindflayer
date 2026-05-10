---
Data: 2026-05-05T00:00:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# (Convex) Non Differentiable Functions
When working with nondifferentiable functions, we lose the concept of a "gradient" (the unique derivative) at kinky points. For [[Convex Functions]], mathematics solves this problem by generalizing the concept of a tangent through subgradients.

### The Concept of a Subgradient
For a smooth convex function, the tangent line always lies "below" the function. In the case of nonsmooth functions (e.g., the absolute value at the origin), there are **infinitely many lines that can pass through the kink** and remain "below" the graph.

A vector $s$ is defined as a **subgradient** of a convex function $f$ at point $x$ if it satisfies the following global inequality for every $z \in \mathbb{R}^{n}$:
$$f(z) \ge f(x) + \langle s, z-x \rangle$$

- **Explanation of the formula:** The right-hand term, $f(x) + \langle s, z-x \rangle$, is the equation of a hyperplane passing through $(x, f(x))$ with "slope" $s$. The inequality tells us that the function $f(z)$will never drop below this hyperplane at any point $z$ in space.
    
- **Too much information:** As the text points out, at nondifferentiable points, there is no "lack" of first-order information, but rather _too much of it_. For the same point $x$, there are infinitely many valid vectors $s$.
    
- **Boundary behavior:** If the point $x$ is "on the border" of the function's domain ($dom(f)$), the norm of the subgradient tends to infinity ($||s||\rightarrow\infty$). This means the supporting hyperplane becomes practically vertical.

![[Pasted image 20260509142616.png | 300]]

![[Pasted image 20260509142645.png | 300]]

![[Pasted image 20260509142700.png | 300]]

![[Pasted image 20260509142751.png| 300]]

![[Pasted image 20260509142843.png | 300]]

### The Subdifferential ($\partial f(x)$)
The set of all valid subgradients at a point $x$ is called the **subdifferential**, denoted by $\partial f(x)$:
$$\partial f(x) = \{s \in \mathbb{R}^{n} : s \text{ is a subgradient at } x\}$$
- **Fundamental properties:** $\partial f(x)$ is a compact convex set for every $x$.
- **Relation to the classic gradient:** If the function $f$ is differentiable at point $x$, the subdifferential restricts to a single element: the classic gradient. Mathematically: $\partial f(x) = \{\nabla f(x)\}$.

![[Pasted image 20260509144117.png | 300]]

![[Pasted image 20260509144137.png | 300]]

How do we know if we are at a minimum point, and how do we move downwards?

- **Global Minimum Condition:** In classic calculus, a point is stationary if $\nabla f(x) = 0$. In the nonsmooth world, a point $x$ is a local (and therefore global) minimum if the zero vector belongs to the subdifferential:
$$0 \in \partial f(x)$$
    Graphically, this means the flat hyperplane ($s=0$) passes exactly through $(x, f(x))$ and lies below the entire function.
    
- **Descent Direction:** A direction $d$ is a "descent direction" if and only if it forms an obtuse angle with _all_ possible subgradients at $x$:
$$\langle s, d \rangle < 0 \quad \forall s \in \partial f(x)$$
    If even one subgradient points in the wrong direction, $d$ does not guarantee a certain descent.
    
- **Steepest Descent:** This is given by the negative of the subgradient with the minimum norm. It is calculated by solving a minimization problem over the set $\partial f(x)$:
$$d^* = -\text{argmin} \{ ||s|| : s \in \partial f(x) \}$$

### Behavior in $\mathbb{R}^n$ and "Pointing Towards the Optimum"
The text presents a graphical example in space: $f(x_{1},x_{2})=max\{x_{1}^{2}+(x_{2}-1)^{2}, x_{1}^{2}+(x_{2}+1)^{2}\}$ with a minimum at $x_* = [0,0]$. At kink points, there are many different $-g$ vectors. Not all of them are descent directions. 

![[Pasted image 20260509144537.png | 300]]

 if $\partial f(x) = \{g=\nabla f(x) \}, g \perp S(f, f(x))$.

![[Pasted image 20260509144641.png | 300]]

However, there is a fundamental property that is vital for the stability of these algorithms: **Any negative subgradient points towards the minimum $x_*$.** (in this case $-g$ points towards $x_*$)

![[Pasted image 20260509144825.png | 300]]

![[Pasted image 20260509144934.png | 300]]

From the definition of a subgradient evaluated at the minimum point $z = x_*$:
$$f(x_*) \ge f(x) + \langle g, x_* - x \rangle$$
Rearranging the terms yields the vital relationship:
$$\langle g, x_* - x \rangle \le f(x_*) - f(x) \le 0$$
This formula means that the dot product between the subgradient $g$ and the vector pointing to the optimum $(x_* - x)$ is always negative or zero. 

Consequently, the opposite of the subgradient ($-g$) forms an acute angle with the ideal direction to reach the minimum. This guarantee is "enough for gradient-type approaches" to work, even if they might lack efficiency.

![[Pasted image 20260509145507.png | 300]]

![[Pasted image 20260509145519.png | 300]]

![[Pasted image 20260509145534.png | 300]]

![[Pasted image 20260509145546.png | 300]]


### Subdifferential Calculus
Just as there are classic differentiation rules (chain rule, derivative of a sum, etc.), there are rules for "computing" $\partial f(x)$ for complex functions.

Here are the operational rules provided:

- **i. Linear combination:** If we scale and add functions by positive constants $\alpha, \beta \in \mathbb{R}_+$, the subdifferentials combine linearly:
    
    $$\partial[\alpha f + \beta g](x) = \alpha \partial f(x) + \beta \partial g(x)$$
    
- **ii. Affine Composition (Pre-composition):** If we apply a linear transformation $Ax+b$ to the domain, we must multiply the subdifferential by the transposed matrix $A^T$:
    
    $$\partial[f(Ax+b)] = A^T[\partial f](Ax+b)$$
    
- **iii. Chain rule (Post-composition):** If $g: \mathbb{R} \rightarrow \mathbb{R}$ is an increasing convex function, the subdifferential of the composed function is the product of the subdifferentials:
    
    $$\partial[g(f(x))] = [\partial g](f(x))[\partial f](x)$$
    
- **iv. Maximum of functions:** If $f(x) = \max\{f_1(x), ..., f_m(x)\}$, the subdifferential is the convex hull (`conv`) of the union of the subdifferentials of the _active_ functions at that point (the set $I(x)$ of functions that "win" the maximum at $x$):
    
    $$\partial f(x) = \text{conv}(\cup_{i \in I(x)} \partial f_i(x))$$
    
- **v. Partial minimization:** If we minimize a function $g(x,y)$ with respect to $y$ only, yielding $f(x)$, the subgradients of $f$ are the "x-components" of the subgradients of $g$ that have zero as their "y-component":
    
    $$\partial f(x) = \{s \in \mathbb{R}^n : (s,0) \in \partial g(x,y)\}$$
    
- **vi. Infimal Convolution:** This is an operation defined as $f(x) = \inf\{f_1(x_1) + f_2(x_2) : x_1 + x_2 = x\}$. Its subdifferential is obtained by intersection:
    
    $$\partial f(x) = \partial f_1(x_1) \cap \partial f_2(x_1)$$
    
    (where $x_1 + x_2 = x$ and $f(x) = f_1(x_1) + f_2(x_2)$).

# References