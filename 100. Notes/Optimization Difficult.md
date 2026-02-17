---
Data: 2026-02-17T16:10:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Optimization Difficult
Before diving into solving optimization problems, it is crucial to understand why they can be inherently difficult, even in the simplest univariate case.

### What if the minimum does not exist ($f_* \nexists$)?
It is possible that the function $f$ does not have a finite global minimum.

![[Pasted image 20260217162046.png | 400]]

* **Unbounded problem:**
    If $f$ has no minimum because it goes to $-\infty$, the problem $(P)$ is said to be **unbounded below**.
$$f_* = \nu(P) = -\infty$$
    
- **Mathematical shorthand:** This is a **convenient shorthand** for saying that for any target value $t$ we can find an input $x$ that gives a result lower than $t$:
$$\forall t \in \mathbb{R} \quad \exists x \in \mathbb{R} \text{ s.t. } f(x) \le t$$
	(i.e., "there is no (finite) lower bound on $im(f)$").
	
-  **Solving (P) involves two tasks:**
    Finding the solution involves distinguishing between two very different outcomes:
    1.  Finding $x_*$ and **proving it is optimal** (how do we verify it is the absolute lowest point?).
    2.  **Constructively proving $f$ is unbounded below** (how do we demonstrate it goes to $-\infty$?).

- **Relevance in Learning:**
    This hardly ever happens in machine learning contexts because loss functions (like Mean Squared Error or Cross-Entropy) are usually bounded below by 0 ($\mathcal{L}(w) \ge 0$).
    However, it is a **nontrivial and important concept in general optimization** theory, closely tied to duality and nonemptiness.

### What if the optimal value exists ($f_* \exists$) but the solution does not ($x_* \nexists$)?
There are cases where the function has a finite lower bound (infimum), but it never actually reaches that value.

* **Asymptotic behavior (e.g., $f(x) = e^x$):**
    The image of the function $im(f) = (0, +\infty)$ is bounded below (by 0), but it has no minimum. The value gets arbitrarily close to 0 as $x \to -\infty$, but never touches it.

![[Pasted image 20260217162416.png | 400]]

* **Discontinuous behavior (e.g., "forcibly"):**
    Consider a function with a discontinuity at the minimum:
    $$f(x) = \begin{cases} 1 & x = 0 \\ |x| & x \neq 0 \end{cases}$$
    Here, the function approaches 0 as $x \to 0$, but at exactly $x=0$, it jumps up to 1.
    *   The **infimum** exists: $\inf \{ f(x) : x \in \mathbb{R} \} = 0$.
    *   The **minimum** does not exist: $\min \{ f(x) : x \in \mathbb{R} \} \nexists$.
    *   Arguably $f_* = 0$, but $\nexists x_*$ s.t. $f_* = f(x_*)$.

![[Pasted image 20260217162501.png | 400]]

* **Topological reason:** This happens when $im(f)$ is an **open set** (or does not contain its boundary).

**Nore**: we use $inf$ instead of $min$ because the first one is used to include also function that set to 0 but never touch it.

### Mathematically Speaking: Infima, Suprema and Extended Reals
To handle these cases rigorously, we look at set theory properties in $\mathbb{R}$.

*   **Total Order:** $\mathbb{R}$ is totally ordered $\implies \forall x, y \in \mathbb{R}$, at least one among $x \le y$ or $y \le x$ holds.
*   **Definitions:**
    *   $\underline{s} = \inf S \iff \underline{s} \le s \ \forall s \in S \land \forall t > \underline{s} \ \exists s \in S \text{ s.t. } s \le t$.
    *   $\bar{s} = \sup S \iff \bar{s} \ge s \ \forall s \in S \land \forall t < \bar{s} \ \exists s \in S \text{ s.t. } s \ge t$.
*   **Relation to min/max:**
    *   If $\underline{s} \in S \implies \underline{s} = \min S$.
    *   If $\bar{s} \in S \implies \bar{s} = \max S$.
*   **Issues:**
    1.  $\inf S / \sup S$ might not exist in $\mathbb{R}$.
    2.  $\inf S / \sup S$ might not belong to the set $S$ (as seen in the $e^x$ example).

*   **Extended Reals ($\overline{\mathbb{R}}$):**
    To fix non-existence, we define $\overline{\mathbb{R}} = \{ -\infty \} \cup \mathbb{R} \cup \{ +\infty \}$.
    Now, for all $S \subseteq \mathbb{R}$, $\exists \sup / \inf S \in \overline{\mathbb{R}}$.
    *   $\inf S = -\infty \iff \forall t \in \mathbb{R} \exists s \in S \text{ s.t. } s \le t$ (unbounded below).
    *   $\inf \emptyset = +\infty, \sup \emptyset = -\infty$.


### Is this a real problem in practice?
In theory, these edge cases are problematic. In computational practice:

1. **Finite Precision:** Computers represent real numbers $x \in \mathbb{R}$ typically as floating point numbers ($x \in \mathbb{Q}$) with limited precision (up to 16 digits). Approximation errors are unavoidable.
2.  **Impossibility of Exactness:** Finding the exact $x_*$ is mathematically impossible in general.
3.  **$\varepsilon$-approximate solutions:** We settle for being "close enough".
    For any fixed $\varepsilon > 0$, we look for $x_\varepsilon \in \mathbb{R}$ such that:
$$f_* \le f(x_\varepsilon) \le f_* + \varepsilon$$
    "As close to the optimal solution (value) as you want."
4.  **Cost:** The cost of solution algorithms typically depends on $\varepsilon$ (sometimes very badly). Additionally, $\varepsilon$ cannot really become infinitely small due to machine precision.

##### Optimization need be approximate
Since we cannot find exact solutions, we define error gaps.

- **Absolute gap:** $A(x) = f(x) - f_* \:\:(\ge 0)$.
- **Relative gap:** $R(x) = \frac{f(x) - f_*}{|f_*|} = \frac{A(x)}{|f_*|} (\ge 0)$.
    * *Why use Relative gap?* It is scale invariant.
    * Consider $(P_\alpha) \min \{ \alpha f(x) \}$. The optimal value scales: $\nu(P_\alpha) = \alpha f_*$.
    * This leads to a different Absolute gap $A(x)$, but the **same** Relative gap $R(x)$.

- **The Problem with Gaps:** Computing $A(x)$ or $R(x)$ requires knowing $f_*$, which is typically unknown (it's what we are trying to find!).
    * One could argue that computing (an estimate of) $f_*$ is "the issue" in optimization.
    * In Machine Learning, sometimes $f_* \approx 0$ is known (e.g., in Neural Networks), but not always (e.g., SVM).

##### Even approximate, optimization is hard / impossible
Why is finding a global minimum so hard, even if we accept approximations?

*   **The Problem of "Needles in a Haystack":**
    Isolated minima can be anywhere. A function could be completely flat everywhere and have a single, deep, narrow "spike" (minimum) at an arbitrary location.

![[Pasted image 20260217164349.png | 400]]

*   **Does restricting to a box $X = [x_-, x_+]$ help?**
    No. Even in a bounded interval, there are uncountably many points.
![[Pasted image 20260217164406.png | 400]]

*   **Is it because the function jumps (discontinuous)?**
	No. Even continuous, smooth functions can have isolated downward spikes that are arbitrarily narrow.
![[Pasted image 20260217164454.png | 400]]

*   **Conclusion:**
    To make (even approximate) optimization possible, the function $f$ **must be "nice"** (e.g., Lipschitz continuous, convex, etc.). Without structure/regularity, optimization is essentially a blind search.
![[Pasted image 20260217164510.png | 400]]

# References