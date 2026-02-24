---
Data: 2026-02-19T16:37:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Linear function

Linear functions are the fundamental building blocks of optimization, where each variable $x_i$ contributes independently and proportionally. A function is linear if it can be expressed as the scalar product between a vector of coefficients $b$ and the input vector $x$:
$$f(x) = \langle b, x \rangle = \sum_{i=1}^n b_i x_i, \quad \text{with } b \in \mathbb{R}^n \text{ fixed.}$$
* **Fundamental Properties:**
    1.  **Homogeneity:** $f(\gamma x) = \gamma f(x)$.
    2.  **Additivity:** $f(x + z) = f(x) + f(z)$.
* **Composition:** It can be seen as the sum of $n$ univariate linear functions $f_i(x_i) = b_i x_i$.

Geometric Interpretation:
* **Graph:** Represents a **hyperplane** in $\mathbb{R}^{n+1}$ (a plane in $\mathbb{R}^3$).
* **Level Sets:** These are parallel hyperplanes in $\mathbb{R}^n$ (lines in $\mathbb{R}^2$) that are all **perpendicular to the vector $b$**.
$$f(x) = f(z) \iff \langle b, z - x \rangle = 0 \iff b \perp z - x$$
![[Pasted image 20260218142222.png]]

##### Tomography & Optimization of Linear Multivariate Functions
We analyze the behavior of the linear function $f(x) = \langle b, x \rangle$ by restricting it to a line. Given a starting point $\bar{x} = 0$ and a direction $\|d\| = 1$, the function along the line is:
$$
\phi(\alpha) = \alpha \langle b, d \rangle = \alpha \|b\| \cos(\theta)
$$
![[Pasted image 20260217214736.png | 500]]

![[Pasted image 20260217215041.png | 500]]

The slope depends on the angle $\theta$ between the gradient $b$ and direction $d$.
- **Increasing:** Occurs if "b is in the same direction as d" (positive correlation):
    - "More collinear" (angle approaches 0) $\implies$ **steeper** (higher rate of growth).
    * Perfectly collinear $\implies$ **maximum slope** (direction of steepest ascent).
    * "Less collinear" $\implies$ **less steep**.
![[Pasted image 20260217214547.png | 500]]
![[Pasted image 20260217215100.png | 500]]

![[Pasted image 20260217220214.png | 500]]

![[Pasted image 20260217220232.png | 500]]

![[Pasted image 20260217220249.png | 500]]
![[Pasted image 20260217220333.png | 500]]

 - **Stationary:**
    * "Flat" (constant value) if $d \perp b$ (the direction is orthogonal to the gradient).   
![[Pasted image 20260217214830.png | 500]]
![[Pasted image 20260217215148.png | 500]]


- **Decreasing:** Occurs if "b is in the opposite direction of d" (negative correlation):
    * "More collinear" $\implies$ **steeper** (faster descent).
    * Perfectly collinear (opposite) $\implies$ **maximum slope (negative)** (direction of steepest descent).
![[Pasted image 20260217214846.png | 500]]
![[Pasted image 20260217215211.png | 500]]


![[Pasted image 20260217220421.png | 500]]

![[Pasted image 20260217220444.png | 500]]


**Global Optimization (Unconstrained)**. The unconstrained minimum is unbounded below: $$f^* = \min \{ f(x) \} = -\infty$$**Exception:** If the gradient $b = 0$, the function is constant everywhere, so $f^* = 0$.
(Note: The same logic applies symmetrically to the global maximum).

**Constrained Optimization (Hyper-rectangle)**
- **Problem:** $\min \{ f(x) : x \in X \}$, where $X$ is a hyper-rectangle (box constraints, $l_i \le x_i \le u_i$).
- **Decomposition:** This multivariate problem splits into $n$ **independent univariate problems**.
	- *Reason:* The linear structure $f(x) = \sum b_i x_i$ implies **separability**; nothing links variable $x_i$ to $x_j$ for $i \neq j$.
    - **Complexity:**
        - The solution consists of $n$ closed-form formulas (checking the sign of $b_i$ against the bounds).
        - Each variable takes $O(1)$ time, making the total complexity $O(n)$.
    - *(Note: This is computationally trivial, "almost like the last time" with univariate linear functions).*


# References