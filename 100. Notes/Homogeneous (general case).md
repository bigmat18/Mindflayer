---
Data: 2026-02-18T16:39:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Homogeneous (general case)

We now remove the assumption of separability. Variables are no longer independent; they interact with each other.

**Definition:** Defined by a fixed matrix $Q \in \mathbb{R}^{n \times n}$ (where $Q_i$ are its rows/columns): 
$$f(x) = \frac{1}{2} x^T Q x = \frac{1}{2} \sum_{i=1}^n q_{ii} x_i^2 + \sum_{i=1}^n \sum_{j=1, j \neq i}^n q_{ij} x_i x_j$$
- **The factor $\frac{1}{2}$:** It is included for mathematical convenience. When we calculate the gradient (derivative), the exponent "2" comes down and cancels with the $\frac{1}{2}$, leaving a clean $Qx$.
- **Coupling:** The terms where $j \neq i$ ($q_{ij} x_i x_j$) link the variables together. You cannot optimize $x_i$ without considering $x_j$.

**Note**: the homogeneous is only a special case where $Q$ is diagonalizzabile.

###### Not Linear
Unlike linear functions, the function of a sum is not the sum of the functions. $$f(x + z) = \frac{1}{2}(x + z)^T Q (x + z) = f(x) + f(z) + z^T Q x$$*Explanation:* The term $z^T Q x$ is the "cross term" or interference between the two vectors $x$ and $z$. If it were linear, this term wouldn't exist.
###### Symmetry Assumption (W.l.o.g.)
We can always assume $Q$ is **symmetric** ($Q = Q^T$).
- **Reason:** $x^T Q x$ is a scalar. Any non-symmetric part of $Q$ cancels out in the sum. We can mathematically replace any $Q$ with its symmetric part $\frac{Q + Q^T}{2}$ without changing the function values.
- **Centering:** Since $f(x)$ contains only quadratic terms (like $x^2$), it is symmetric with respect to the origin: $f(x) = f(-x)$. The "center" of the bowl/saddle is always at $x=0$.

###### Tomography (Behavior along a line)
If we slice the function along a direction $d$ (passing through 0), we get a univariate quadratic function:    
$$\phi(\alpha) = f(\alpha d) = \frac{1}{2} \alpha^2 (d^T Q d)$$
- *Interpretation:* The shape is a parabola starting at 0.
- **Curvature:** The term $(d^T Q d)$ determines if the parabola opens Up (convex) or Down (concave).
	- If $d^T Q d > 0$: It goes up (Smile shape $\cup$).
    - If $d^T Q d < 0$: It goes down (Frown shape $\cap$).
    - The steepness depends on the magnitude of $d^T Q d$.

###### Spectral Decomposition ([[Eigenvalues and Eigenvectors]])
To understand the shape fully, we look at the eigenvalues of $Q$.
$$Q = H \Lambda H^T = \sum_{i=1}^n \lambda_i h_i h_i^T$$

we use the **spectral decomposition**. 

- $\lambda_i$ (Eigenvalues): Represent the curvature along the principal axes.
    - $\lambda_{max} = \max \{ d^T Q d : \|d\|=1 \}$ (Steepest ascent direction).
    - $\lambda_{min} = \min \{ d^T Q d : \|d\|=1 \}$ (Flattest or steepest descent direction).
- $h_i$ (Eigenvectors): Represent the directions of the principal axes (the axes of the ellipsoid).

The behavior of the function depends entirely on the signs of the eigenvalues ($\lambda_i$):
* **Positive Definite ($Q \succ 0$):** All $\lambda_i > 0$.
    * Shape: A "bowl" or valley.
    * Global Minimum at $x=0$.
* **Positive Semidefinite ($Q \succeq 0$):** All $\lambda_i \ge 0$.
    * Shape: A "trough" (flat bottom in some directions).
    * Infinitely many global minima (a valley floor).
* **Indefinite:** Some $\lambda_i > 0$, some $\lambda_i < 0$.
    * Shape: A **Saddle Point**. It curves up in some directions and down in others.
    * Unbounded below ($-\infty$).
* **Negative Definite ($Q \prec 0$):** All $\lambda_i < 0$.
    * Shape: A "hill".
    * Global Maximum at $x=0$ (unbounded below).

### Tomography of Homogeneous Quadratic Functions
We analyze a specific numerical example to see how the eigenvalues ($\lambda$) and eigenvectors ($H_i$) determine the shape of the function.

**Fundamental relation:** Along the direction of an eigenvector $H_i$ (where $\|H_i\|=1$), the function becomes a simple parabola determined solely by the eigenvalue $\lambda_i$:
$$
    \phi_{H_i}(\alpha) = f(\alpha H_i) = \frac{1}{2} \alpha^2 \lambda_i \quad
 $$
- *Explanation:* If you move along an eigenvector, the matrix $Q$ acts like a scalar multiplier $\lambda_i$. The cross-terms vanish, and you are left with a univariate quadratic scaled by $\lambda_i$.

 **Classification of Behavior along $d$:**  The shape of the parabola $\phi(\alpha)$ depends entirely on the sign of the scalar value $d^T Q d$:

 1.  **Strictly Convex ($d^T Q d > 0$):**
    * The parabola opens **upward** (like a smiley face $\cup$).
    * $f$ grows to $+\infty$ as you move away from the origin along $d$.
    * Interpretation: You are ascending the side of a valley.
    * As the direction $d$ rotates, the parabola remains U-shaped but changes width. It is **narrowest (steepest)** along the eigenvector with the largest $\lambda$, and **widest (flatter)** along the eigenvector with the smallest $\lambda$.

![[Pasted image 20260218154734.png | 500]]
![[Pasted image 20260218154523.png | 500]]


2.  **Positive Semidefinite / Not Strictly Convex ($d^T Q d \geq 0$ but $\exists d \:\:t.c.\:\: d^TQd = 0$):** 
	- This corresponds to a **valley** or "trough" shape. 
	- There is a direction where the function does not grow (it stays flat).

![[Pasted image 20260218154810.png | 500]]
![[Pasted image 20260218154820.png | 500]]
![[Pasted image 20260218154853.png | 500]]
![[Pasted image 20260218155434.png | 500]]
![[Pasted image 20260218155456.png | 500]]


3. **Can be both ($d^T Q d < 0 \text{ and } d^T Q d > 0$)**:
* Steepest negative slope is along $H_2$ ($\lambda_2 < 0$). 
* Steepest positive slope is along $H_1$ ($\lambda_1 > 0$).

![[Pasted image 20260218154638.png | 500]]
![[Pasted image 20260218154649.png | 500]]
![[Pasted image 20260218155629.png | 500]]
![[Pasted image 20260218155735.png | 500]]

3. **Strictly Concave ($d^T Q d < 0$):**
    * The parabola opens **downward** (like a frown $\cap$).
    * $f$ goes to $-\infty$ as you move away from the origin along $d$.
    * Interpretation: You are descending the side of a hill.
    * There is no minimum in this function (only a global maximum at 0).
    
![[Pasted image 20260218155807.png | 500]]
![[Pasted image 20260218155823.png | 500]]
![[Pasted image 20260218155836.png | 500]]

### Homogeneous quadratic functions: graph and level sets
We analyze the full geometry of the function $f(x) = \frac{1}{2} x^T Q x$ by looking at its **graph** (surface in 3D) and its **level sets** (contours in 2D).

**Classification of Shapes:**
1.  **Elliptic / Positive Definite ($Q \succ 0$, all $\lambda_i > 0$):**
    * **Graph:** An elliptic **paraboloid** (a "bowl").
    * **Level Sets:** **Ellipses** centered at the origin.
    * **Optimization:** Unique global minimum at $x=0$.
	- The "narrowness" of the ellipses is determined by $\lambda$. Large $\lambda$ (steep curvature) $\implies$ short axis (narrow ellipse). Small $\lambda$ (flat curvature) $\implies$ long axis (wide ellipse).

![[Pasted image 20260218161352.png | 600]]


2.  **Parabolic Cylinder / Positive Semidefinite ($Q \succeq 0$, some $\lambda_i = 0$):**
    * **Graph:** A **valley** or "trough" (like a half-pipe).
    * **Level Sets:** **Parallel lines** (degenerate ellipses that extend to infinity).
    * **Optimization:** Infinitely many global minima along the line/subspace corresponding to the zero eigenvalue.
	- Along the direction of the eigenvector with $\lambda=0$, the graph is perfectly flat (a straight line). In orthogonal directions, it curves up like a parabola.

![[Pasted image 20260218161407.png | 600]]


3.  **Hyperbolic / Indefinite (some $\lambda_i > 0$, some $\lambda_j < 0$):**
    * **Graph:** A **hyperbolic paraboloid** (a "saddle" or Pringles chip).
    * **Level Sets:** **Hyperbolas**.
    * **Optimization:** No minimum or maximum. The point $x=0$ is a **Saddle Point** (stationary but not an extremum).
	- There are two distinct sectors. In one direction (positive $\lambda$), the level sets curve inward (valley). In the other (negative $\lambda$), they curve outward (hill). The lines separating these sectors represent directions of zero curvature.

![[Pasted image 20260218161457.png | 600]]


4.  **Inverted Elliptic / Negative Definite ($Q \prec 0$, all $\lambda_i < 0$):**
    * **Graph:** An inverted paraboloid (a "hill").
    * **Level Sets:** Ellipses.
    * **Optimization:** Unique global **maximum** at $x=0$. Unbounded below ($-\infty$).
	- Identical to the Positive Definite case, but the $z$-axis is flipped. The ellipses represent contours of decreasing height.
	
![[Pasted image 20260218161522.png | 600]]

### Optimizing a homogeneous quadratic multivariate function 
 Clearly depends sign of eigenvalues of $Q \equiv$ definiteness:
- $Q \succeq 0 \land Q \preceq 0 \equiv \lambda_1 = \lambda_n = 0 \equiv Q = 0 \implies \min = \max = 0$ (constant)
- $Q \succeq 0 \implies \min = 0, \text{argmin} = 0, \max = +\infty$
- $Q \preceq 0 \implies \max = 0, \text{argmax} = 0, \min = -\infty$
- $Q \succ \prec 0 \implies \max = +\infty, \min = -\infty$
analogous to univariate case, but "many more ways to be $> 0 / < 0$"

Box-constrained optimization on (closed) hyperrectangle $X$ absolutely not analogous to the univariate case:
- $NP$-hard in most cases
- $\min$ with $Q \succeq 0$ and $\max$ with $Q \preceq 0$ **polynomial** but **nontrivial**
- $NP$-hardness due to $\mathbb{R}^n$ "big" ($X$ has $2^n$ vertices), issue also in $P$ case
- $\max\{ f(x) \}$ and $\min\{ f(x) \}$ very very different
# References