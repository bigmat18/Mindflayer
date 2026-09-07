---
Data: 2026-02-17T16:08:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Optimization Problems

### Functions and Sets
We begin by defining a generic function $f: \mathbb{R} \to \mathbb{R}$, where $x$ is the input space and $y = f(x)$ is the output.

- **Graph ($gr(f)$):** The set of input-output pairs in the Cartesian space $\mathbb{R}^2$.
$$gr(f) = \{ (f(x), x) : x \in \mathbb{R} \} \subset \mathbb{R}^2$$
![[Pasted image 20260217145728.png | 400]]

- **Image ($im(f)$):** The set of all values that the function can take (also called the effective co-domain). It is the projection of the graph onto the output space.
$$im(f) = \{ y : \exists x \in \mathbb{R} \text{ s.t. } y = f(x) \} \subset \mathbb{R}$$
![[Pasted image 20260217145747.png | 400]]

- **Level Set ($L(f, v)$):** The set of points in the domain that return the same value $v$. 
$$L(f, v) = \{ x \in \mathbb{R} : f(x) = v \} \subset \mathbb{R}$$
![[Pasted image 20260217145819.png | 400]]

The roots of the function correspond to $L(f, 0)$.
![[Pasted image 20260217145943.png | 400]]


## The Optimization Problem
The core of the course is solving the minimization problem (without loss of generality, since $\max f = - \min (-f)$).

### Problem Definition (P)
f objective (function) of (univariate, unconstrained) optimization problem
$$(P) \quad f_* = \min \{ f(x) : x \in X \}$$
Where $X \subseteq \mathbb{R}$ is the **feasible region**.

*   **Optimal Value ($f_*$):** The smallest value assumed by the function within the set $X$.
$$f_* = \min( im(X, f) )$$
    Alternatively:  $f_*$ = smaller element of $im(f)$ = smaller $v$ s.t. $L(f, v) \neq 0$ 
    - that means $f_*$ is the smallest $v$ such that the level set intersects the feasible region ($L(f, v) \cap X \ne \emptyset$)

*   **Optimal Solution ($x_*$):** The point (or points) where the function reaches the minimum.
$$(P) \quad x_* \in \text{argmin} \{ f(x) : x \in X \}$$
$x_*$ s.t. $f(x*) \leq f(x) \forall x \in \mathbb{R}$ optimal solutions (if $\exists$, which it may not)

![[Pasted image 20260217151211.png | 400]]

$x_*$ may no be unique: $\exists x' \neq x_* \in L(f, f_*) = X_*$ **set of optimal solutions**. but we don’t care (mostly): all optimal solutions equivalent “in the eyes of f"

![[Pasted image 20260217151635.png | 400]]

### Reformulations
Sometimes changing the objective function simplifies the problem without changing the position of the minimum ($X_*$ remains unchanged):
$$min \{ f(x) : x\in \mathbb{R} \} = -\text{max} \{ -f(x) : x\in \mathbb{R} \} \:\:\: \text{ i.e. } \:\:\: argmin\{f(x) \in \mathbb{R}\} = argmax\{-f(x) : x\in \mathbb{R}\}$$
but  $min\{ f(x)\} \neq \max\{f(x)\}$, often **rather different** problems

![[Pasted image 20260217151818.png | 450]]

Analogously **translation**:
$$
min \{ f(x) + c : x\in \mathbb{R} \} = c + min \{ f(x) : x\in \mathbb{R} \} \:\:\: \text{ i.e. } \:\:\: argmin\{f(x) + c \in \mathbb{R}\} = argmin\{f(x) : x\in \mathbb{R}\}
$$
![[Pasted image 20260217152601.png | 450]]

Analogously **scaling**:
$$
min \{ cf(x) : x\in \mathbb{R} \} = c \cdot min \{ f(x) : x\in \mathbb{R} \} \:\:\: \text{ i.e. } \:\:\: argmin\{cf(x) \in \mathbb{R}\} = argmin\{f(x) : x\in \mathbb{R}\}
$$
![[Pasted image 20260217152748.png | 450]]


While the unconstrained problem seeks the minimum over the entire real line $\mathbb{R}$, the general case involves restricting the search to a specific subset.

### (Univariate) Constrained optimization problem 
Let $X \subseteq \mathbb{R}$ be any set, defined as the **feasible region**. The constrained optimization problem is formulated as:
$$(P) \quad f_* = \min \{ f(x) : x \in X \}$$
Where:
*   $f: X \to \mathbb{R}$ is the objective function.
*   **Feasible solution:** A point $x \in X$.
![[Pasted image 20260217155231.png | 400]]

*   **Unfeasible solution:** A point $x \in \mathbb{R} \setminus X$.
![[Pasted image 20260217155252.png | 400]]

- **Optimal Value ($f_*$):** It is the smallest element of the image of $X$:
$$f_* = \nu(P) = \min( im(X, f) )$$
![[Pasted image 20260217155428.png | 400]]

  - **Set of Optimal Solutions ($X_*$):**
    It is the intersection between the level set at the optimal value and the feasible region:
$$X_* = L(f, f_*) \cap X = \{ x \in X : f(x) = f_* \}$$
    This represents the set of *best* feasible solutions.

![[Pasted image 20260217155633.png | 400]]

> **Note:** A constraint $X$ can be **"useless"** if $X_*$ coincides with the set of optimal solutions of the unconstrained problem (or partly useless if the value $f_*$ does not change). This justifies the preliminary study of the unconstrained case $X = \mathbb{R}$.

![[Pasted image 20260217155823.png | 400]]

### Specifying the Set $X$
The abstract constraint "$x \in X$" needs to be specified concretely. It is often useful to represent the set $X$ via **one or more auxiliary functions**.

1.  **Equality Constraint:**
    Defined by an equation $g(x) = v$. The feasible set is a level set:
$$X = L(g, v) = \{ x \in \mathbb{R} : g(x) = v \}$$
    *(Typically in $\mathbb{R}$ this reduces $X$ to a discrete set of points).*

![[Pasted image 20260217155938.png | 400]]

2.  **Inequality Constraint:**
    Defined by $g(x) \le v$. The feasible set is a sublevel set:
$$X = S(g, v) = \{ x \in \mathbb{R} : g(x) \le v \}$$
![[Pasted image 20260217160221.png | 400]]
###### Conventions and Standardization
*   **"v hidden in f":** By convention, constants are moved inside the function to always have comparisons with 0.
$$g(x) \le v \implies \tilde{g}(x) = g(x) - v \le 0$$
*   **Sign of the inequality:** If one has a constraint $g(x) \ge 0$, this is transformed into $-g(x) \le 0$ to maintain a standard form.

![[Pasted image 20260217160403.png | 400]]

###### Multiple Constraints
Usually, a problem presents multiple simultaneous constraints (e.g., "$g_1(x) \le 0$ AND $g_2(x) \le 0$").
Mathematically, the logical conjunction ("first condition **and** second condition") corresponds to the **intersection** of the sets:
$$X = \bigcap_i S(g_i, 0)$$
###### Bounds (Interval Constraints)
The most common and simple constraints in univariate optimization are bounds on the variable $x$:

*   **Upper bound:** $x \le x_+$ (Set $X = (-\infty, x_+]$).
![[Pasted image 20260217160526.png | 400]]


*   **Lower bound:** $x \ge x_-$ (Set $X = [x_-, +\infty)$).
![[Pasted image 20260217160642.png | 400]]

*   **Box (Closed interval):** The intersection of an upper bound and a lower bound defines a "box":
    $$X = [x_-, x_+] = \{ x \in \mathbb{R} : x_- \le x \le x_+ \}$$
![[Pasted image 20260217160714.png | 400]]

> **Important:** In practical applications and algorithms, it is preferred to work with closed sets (like $[x_-, x_+]$) rather than open sets (like $(x_-, x_+)$), to ensure the existence of the minimum (Weierstrass Theorem) and avoid situations where the infimum exists but the minimum does not.

# References