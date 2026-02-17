---
Data: 2026-02-17T19:34:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Simple Functions, Multivariate case
Moving from a single variable ($x \in \mathbb{R}$) to multiple variables ($x \in \mathbb{R}^n$) is the most significant leap in optimization theory. It transforms a simple "line search" into a search within a potentially infinite-dimensional landscape.

We consider functions $f:\mathbb{R}^{n}\rightarrow\mathbb{R}$. Here, the input is a **vector** $x = [x_1, x_2, \dots, x_n]^T$ representing $n$ different parameters, while the output remains a single **scalar** value (the "cost" or "loss").
* **Visualization:** While we can visualize $n=1$ as a curve and $n=2$ as a surface in 3D, for $n \ge 3$ we lose the ability to "see" the landscape. We must rely entirely on mathematical tools (gradients, Hessians) to navigate.

The difficulty of an optimization problem is often dictated by its dimensionality:
-   **Smallish ($n \le 100$):** Can often be solved with very precise, "heavy" second-order methods (like Newton's method).
-   **Largish ($n \approx 10^5$):** Typical of medium-sized engineering problems or simple machine learning models. Requires efficient first-order methods (Gradient Descent).
-   **Heinously Large ($n \ge 10^9$):** Found in modern **Large Language Models (LLMs)**. At this scale, even storing the coordinates of $x$ in memory is a challenge, and we must use highly optimized distributed systems.

All fundamental concepts ($f_{*}, X_{*}$) generalize mathematically, but the **volume** of the space $\mathbb{R}^n$ expands exponentially.

> "The vector space $\mathbb{R}^{n}$ is big. Really big. You just won't believe how vastly, hugely, mind-bogglingly big it is." — *Adapted from Douglas Adams*

**Why is it harder?**
In $\mathbb{R}^1$, you only have two directions to move (left or right). In $\mathbb{R}^n$, there are infinitely many directions. Finding the "downward" path becomes a needle-in-a-haystack problem unless the function has a specific structure (like convexity).

Often, we restrict our search to a feasible region $X$. The simplest multivariate constraint is a **hyperrectangle** (or "box"):
$$X=\{x\in\mathbb{R}^{n}:x_{-}\le x\le x_{+}\}, \quad x_{\pm}\in\mathbb{R}^{n}$$
* **The Combinatorial Explosion:** Consider a simple case where each variable $x_i$ can only be $0$ or $1$ (a **binary hypercube**). 
    * If $n=10$, we have $2^{10} = 1,024$ points (manageable).
    * If $n=100$, we have $2^{100} \approx 1.26 \times 10^{30}$ points.
    
Even if the space is "bounded" and "small" (between 0 and 1), the number of possible configurations is so vast that a brute-force search is impossible. This is why multivariate optimization requires sophisticated algorithms that don't check every point, but instead "feel" the slope of the landscape.

### An aside: Vector-valued optimization ($f:\mathbb{R}^{n}\rightarrow\mathbb{R}^{k}$)
How about $f:\mathbb{R}^{n}\rightarrow\mathbb{R}^{k}$? Already $f:X\rightarrow\mathbb{R}$ is a rather strong assumption: it implies we can "express all the value of any $x\in X$ with a single number". Given $x^{\prime}$ and $x^{\prime\prime}$ we can always tell which one I like best ($\mathbb{R}$ has total order).

* **The Problem:**
    Often there would be more than one objective:
    $$(P) \quad \min\{[f_{1}(x),f_{2}(x),...]:x\in X\}$$
    with $f_{1}$, $f_{2}$,... contrasting and/or with incomparable units (apples vs. oranges).  Examples:
    - Car cost vs. flashiness vs. km/l; 
    - Loss function $\mathcal{L}(w)$ vs. regularity $R(w)$ in ML.

###### Example: Portfolio Selection (Markowitz Model)
This is the classic "Return vs. Risk" trade-off:
- **$X$**: All possible combinations of stocks/bonds you can buy.
- **$f_1(x)$**: Expected Return (Maximize).
- **$f_2(x)$**: Risk/Volatility (Minimize).

![[Pasted image 20260217202710.png | 400]]

Since we can't find a single "best" solution, we look for **non-dominated solutions**.
* **Domination**: A solution $x_A$ dominates $x_B$ if $x_A$ is better or equal in *all* objectives and strictly better in at least one.
* **Pareto Frontier**: The set of all solutions that are not dominated by any other. On this frontier, you cannot improve one objective without worsening another.

![[Pasted image 20260217202904.png | 400]]

To actually solve the problem, we usually "transform" it back into a single-objective problem using two main strategies:
1. **Scalarization (Weighted Sum)**: In the first, We combine all objectives into a single formula using weights ($\alpha$):
$$(P_\alpha) \quad \min f_1(x) + \alpha f_2(x)$$
	* **The logic:** You decide how much 1 unit of "risk" is worth in terms of "return."
	* **The issue:** How do you choose $\alpha$? The "best" $\alpha$ is often subjective and requires a "divine" insight or a grid search.

![[Pasted image 20260217202931.png | 400]]

2. **Budgeting ($\epsilon$-Constraint Method)**: We pick one primary objective to optimize and turn the others into constraints (budgets):
$$(P_\beta) \quad \min f_2(x) \quad \text{subject to } f_1(x) \ge \beta_1$$
	* **The logic:** "Give me the minimum risk, provided that the return is at least 5%."
	* **The issue:** How do you choose the threshold $\beta_1$? Setting it too high might make the problem impossible (infeasible).

![[Pasted image 20260217202946.png | 400]]

![[Pasted image 20260217204048.png | 400]]

> **Note:** In Machine Learning, we see this constantly with **Regularization**. We minimize the Error $\mathcal{L}(w)$ while "budgeting" or "weighting" the complexity of the model $R(w)$ (e.g., LASSO or Ridge regression).

### Picturing multivariate functions (Tomography)
Visualizing the graph of a function $gr(f) \in \mathbb{R}^{n+1}$ becomes impossible as soon as $n > 2$ (which would require a 4D plot). To study these functions, we use **Tomography**: instead of looking at the whole landscape, we look at a "slice" along a line.

* **The Concept:** Given a point $x$ and a direction $d$, we define a univariate function $\phi$ that represents the values of $f$ as we move along the line passing through $x$ in direction $d$.
    $$\phi_{x,d} ( \alpha ) = f ( x + \alpha d ) : \mathbb{R} \rightarrow \mathbb{R}$$
* **Key Properties:**
    * **Scale:** Changing the length of $d$ ($\| d \|$) only stretches or compresses the graph of $\phi$. For this reason, we usually use a **normalised direction** ($\| d \| = 1$).
    * **Coordinate Restriction:** The simplest tomography is moving along a single axis (the $i$-th coordinate). This means varying $x_i$ while keeping all other $x_j$ constant. This is the basis for partial derivatives.
    * **Utility:** While we can't see the whole function, we can always plot $gr( \phi_{x,d} )$ on a 2D plane to understand the behavior of $f$ locally.

### The Simplest Multivariate Functions: Linear 
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
![[Pasted image 20260217212859.png | 550]]

### Tomography & Optimization of Linear Multivariate Functions
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


### A very simple quadratic function: separable (non-homogeneous) 


# References