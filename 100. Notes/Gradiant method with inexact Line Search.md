---
Data: 2026-03-24T20:57:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Smooth Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Gradient Method with Inexact Line Search
When tackling the optimization of general functions, finding the **exact optimal step size along a descent direction** (Exact Line Search) is, in most cases, impossible or computationally too **expensive**. 

Standard stopping criterion for [[Gradient Method|Line Search]] is $|\varphi'(\alpha)| \leq \epsilon'$ but $\epsilon' = 0$ (exact) in general not possible. We choose it only (approzimate) stationary point of $\varphi$ needed.
- Good News: the alghorithm work with $\epsilon'_i = \epsilon ||\nabla f(x^i)||$. What is a simple prove:
$$
\{x^i\} \to x \Longrightarrow ||\nabla f(x)|| \leq \epsilon
$$
if it converges, then it does at an (approximate) stationary point. Proof:
$$
\{x^i\} \to x \text{ and } |\varphi'(\alpha^i)| \leq \epsilon_i' \forall i \text{ and } f\in C^1 \equiv \nabla f \in C^0 \Longrightarrow
$$
$$
\lim_{i \to \infty} |<\nabla(x'), \nabla f(x^{i+i})>| \leq \lim_{i\to\infty} \epsilon_i^i
$$
$$
\equiv <\nabla f(x), \nabla f(x)> \leq \epsilon ||\nabla f(x)|| \Longrightarrow ||\nabla f(x)|| \leq \epsilon
$$
- Bad News: the LS should become more accurate as the algorithm proceeds down to $\epsilon = \epsilon^2$ this means rather **high** accuracy

For this reason, we resort to **Inexact Line Search** procedures, where we do not look for the perfect minimum, but settle for a step $\alpha$ that decreases the function "enough".

## The Armijo Condition
The first problem to solve is preventing the algorithm from taking steps that are too long, which could cause it to diverge. To guarantee a sufficient decrease (this define how "enagh decrease" means), we introduce the **Armijo condition**, based on a parameter $0 < m_1 < (\ll) 1$:

![[Pasted image 20260324215835.png | 350]]

$$(A) \quad \varphi(\alpha) \le \varphi(0) + m_1 \alpha \varphi'(0)$$
- $\varphi(\alpha)$ represents the value of our function moving by a step $\alpha$ along the descent direction.

- $\varphi(0) + m_1 \alpha \varphi'(0)$ is the equation of a line (a first-order model) passing through the starting point. The original slope $\varphi'(0)$ is "relaxed" by multiplying it by a very small fraction $m_1$ (often $m_1 \approx 0.0001$).

![[Pasted image 20260324215914.png | 350]]

In practice, this condition requires that the function decreases by at least a small fraction ($m_1$) of the descent that was "promised" by the initial derivative at the starting point.

If we enforce that the step is never smaller than a certain threshold ($\alpha^i \ge \overline{\alpha} > 0$) and condition (A) holds at every iteration $i$, the algorithm works: every accumulation point of the sequence $\{x^i\}$ is a stationary point (where $\{||\nabla f(x^i)||\} \to 0$), unless the function diverges to $-\infty$. 

![[Pasted image 20260324215953.png | 350]]

In fact, the decrease at each step is guaranteed by the inequality:
$$
-\varphi_i(0) = ||\nabla f(x^i)||^2 \geq \epsilon > 0 \text{ and (A) hold } \forall i \Longrightarrow 
$$
$$f^{i+1} \le f^i + m_1 \alpha^i \varphi_i'(0) \le f^i - m_1 \overline{\alpha} \epsilon \Longrightarrow$$

Applying this logic iteratively yields $f^i \le f^0 - m_1 \overline{\alpha} \epsilon i \Rightarrow \{f^i\} \to -\infty$.

Don't even need $\alpha^i \geq \bar{\alpha} > 0$ just $\sum_{i=1}^{\infty} \alpha^i = \infty$. But how do we ensure that $\alpha^i$ does not get "too small"? We need add some further condition to (A).

## The Wolfe Condition
While Armijo prevents steps that are too long, we need a rule to avoid steps that are _too short_, otherwise the algorithm might stall far from the solution. 

![[Pasted image 20260324233520.png | 350]]

To solve this problem, we introduce a second parameter $m_2$ such that $m_1 < m_2 < 1$ and formulate the **Wolfe Condition:**
$$(W) \quad \varphi'(\alpha) \ge m_2 \varphi'(0)$$
This condition requires the derivative at the new point $\alpha$ to be "closer to 0" (less steep) than the starting derivative. If the slope is still very negative, it means there is still plenty of room to descend and the step taken was too timid.

![[Pasted image 20260324233554.png | 350]]

#### Strong Wolfe Condition
There is also a stricter version, the **Strong Wolfe condition**:
$$(W') \quad |\varphi'(\alpha)| \le m_2 |\varphi'(0)|= -m_2 \varphi'(0) \Rightarrow (W)$$
![[Pasted image 20260324233613.png | 350]]

By taking the absolute value, we force the derivative to be numerically small. The intersection $(A) \cap (W')$ ensures that $\varphi'(\alpha) \not\gg 0$, helping the algorithm to discard some local maxima and capture the desired local minima. In 0 the wolfe condtion is not soddisfatta.

![[Pasted image 20260324234413.png | 350]]

![[Pasted image 20260324234432.png | 350]]

## Mathematical Proof (Rolle's Theorem)
One might wonder: does there always exist a step $\alpha$ that simultaneously satisfies both Armijo and Wolfe? The answer is yes, and it is proven using Rolle's Theorem.

Assuming $\varphi \in C^1$ and that $\varphi(\alpha)$ is bounded below for $\alpha \ge 0$ , we define the distance between the Armijo limit line $l(\alpha)$ and our function:

$$d(\alpha) = l(\alpha) - \varphi(\alpha)$$
$$d'(\alpha) = m_1 \varphi'(0) - \varphi'(\alpha)$$

We know that $d(0) = 0$. Since the function is bounded below, there will be a smallest point $\overline{\alpha} > 0$ where the curve "touches" the line again, meaning $d(\overline{\alpha}) = 0$. By Rolle's Theorem, if $d(0) = d(\overline{\alpha}) = 0$, there must exist an intermediate point $\alpha' \in (0, \overline{\alpha})$ where the derivative vanishes:

$$d'(\alpha') = 0 \equiv m_1 \varphi'(0) = \varphi'(\alpha')$$

Since we set $m_2 > m_1$, and knowing that $\varphi'(0)$ is negative, we deduce that:

$$m_2 \varphi'(0) < m_1 \varphi'(0) = \varphi'(\alpha')$$

This mathematically proves that the Wolfe condition $(W)$ is satisfied at the point $\alpha'$.

## Backtracking Line Search
As mentioned, finding the exact point $\alpha^*$ that simultaneously satisfies the Armijo and Wolfe conditions can be complicated. One approach involves choosing an $m_1$ "small enough" (often $m_1 = 0.0001$ is sufficient) so that local minima are not excluded, and stopping as soon as a point satisfying $(A) \cap (W) / (W')$ is found. Specialized algorithms exist for more complex cases.

However, there is an even simpler and widely used version in practice: the **Backtracking Line Search (BLS)**. This variant completely ignores the search for the Wolfe condition and limits itself to verifying only the **Armijo condition (A)** by iteratively contracting the step size.

Plaintext

```
procedure alpha = BLS(phi, alpha, m_1, tau) // tau < 1
    while(phi(alpha) > phi(0) + m_1 * alpha * phi'(0)) do 
        alpha <- tau * alpha;
```

- **Logic:** The algorithm assumes an initial large trial step (typically $\alpha = 1$). As long as the Armijo condition is not satisfied (i.e., the function value is greater than the minimum required decrease $m_1 \alpha \varphi'(0)$), the step is contracted by multiplying it by a constant factor $\tau < 1$
    
- **Guarantees:** We know from theory that there is always an interval of valid steps near zero: there exists $\overline{\alpha}^i > 0$ such that condition (A) is satisfied $\forall \alpha \in (0, \overline{\alpha}^i]$.
    
- Since we start from $\alpha = 1$ and continue multiplying by $\tau$, the loop will surely stop. The BLS will produce a final step $\alpha \ge \tau^{h_i}$, where $h_i$ is the minimum number of contractions necessary for the step to enter the valid interval: $h_i \ge \min\{k : \tau^k \le \overline{\alpha}^i\}$.
    
- **Convergence Condition:** If these valid intervals do not shrink to zero asymptotically (i.e., if $\overline{\alpha}^i \ge \overline{\alpha} > 0 \quad \forall i$), then a maximum number of contractions $h$ will exist such that $\alpha \ge \tau^h \quad \forall i$, thereby guaranteeing the convergence of the algorithm.
    

To obtain this guarantee that the interval does not vanish, we need to impose some regularity conditions on the function $f$ (**L-smoothness**), which we will explore below.

## [[Optimization Difficult#Lipschitz Continuity|Lipschitz continuous]] and L-Smoothness
To mathematically guarantee that the calculated step size does not degenerate toward zero, we need to qualify how "smooth" or predictable our function is.
- **Lipschitz Function:** $f$ is (globally) Lipschitz continuous with constant $L$ ($L$-c) if: $$|f(x) - f(z)| \le L\|x - z\| \quad \forall x, z$$The function cannot have vertical jumps; its maximum variation between two points is strictly limited by a constant $L$ proportional to their distance.
    
- If $f \in C^1$, saying that $f$ is $L$-c is equivalent to saying that its gradient is bounded above: $\sup\{\|\nabla f(x)\|\} = L < \infty$ (This can be easily proven using the Mean Value Theorem).
    
- Conversely, if $f$ is globally $L$-c with constant $L$, then $\|\nabla f(x)\| \le L$.

The truly crucial piece for optimization, however, is **L-smoothness**:
- **L-smoothness:** $f$ is $L$-smooth on the set $X$ if its gradient $\nabla f$ is Lipschitz continuous with constant $L$:$$\|\nabla f(x) - \nabla f(z)\| \le L\|x - z\| \quad \forall x, z \in X$$$L$-smoothness means that the slope of our curve (the gradient) never changes abruptly.
    
- The **Hessian matrix** $\nabla^2 f$ is the Jacobian of the gradient. Saying that $\nabla f$ is $L$-c is equivalent to saying that $\nabla^2 f$ is bounded.
    
- If $f \in C^2$ (twice differentiable), saying $f$ is $L$-smooth is equivalent to saying the eigenvalues of the Hessian are trapped between $-L$ and $L$: $$-LI \le \nabla^2 f(x) \le LI \quad \forall x \equiv \max\{|\lambda^1|, |\lambda^n|\} \le L$$
- If $f \in C^2$ is also **convex**, we know the curvatures cannot be negative, so the condition becomes: $$0 \le \nabla^2 f(x) \le LI \equiv 0 \le \lambda^n \le \lambda^1 \le L$$
    (where $\lambda^1$ and $\lambda^n$ are the maximum and minimum eigenvalues, respectively).

## Necessary Technical Results
To link the properties of the $n$-dimensional function $f$ to our one-dimensional function $\varphi(\alpha)$ along the descent direction $d$, we use the chain rule.
- **Technical result for the directional derivative:** $$\varphi_{x,d}'(\alpha) = \frac{\partial f}{\partial d}(x + \alpha d) = \langle\nabla f(x + \alpha d), d\rangle$$
- **Fundamental consequence:** If we evaluate the initial derivative (at $\alpha = 0$) taking the steepest descent direction $d = -\nabla f(x)$, we get: $$\varphi_{x, -\nabla f(x)}'(0) = \langle\nabla f(x), -\nabla f(x)\rangle = -\|\nabla f(x)\|^2 < 0$$
    This mathematically proves that "the further $\nabla f(x)$ is from 0, the steeper (and more negative) $\varphi'(0)$ is.
- If $f$ is $L$-smooth, then the one-dimensional function $\varphi$ is $[L\|d\|^2]$-smooth.
- **Intuitively:** The initial derivative $\varphi'(0)$ starts with a large negative value and, due to the smoothness limited by $[L\|d\|^2]$, it can only decrease slowly. This guarantees that the step size (the step $\alpha$) will never have to become too small to find a valid descent.

## Convergence with Armijo-Wolfe / Backtracking LS
Now we put the pieces together to prove that the step $\alpha$ does not degenerate, guaranteeing convergence.

Recall that $\alpha'$ is the smallest step ($0 < \alpha' < \overline{\alpha}$) for which the **Armijo condition** is valid and the **Wolfe condition** simultaneously triggers: $\varphi'(\alpha') = m_2 \varphi'(0) > \varphi'(0)$.

Since $\varphi$ is $[L\|d\|^2]$-smooth, we can find a lower bound for $\alpha'$ (and consequently for our limit $\overline{\alpha}$ in Backtracking LS): $$L\|d\|^2(\alpha' - 0) \ge \varphi'(\alpha') - \varphi'(0) > (1 - m_2)(-\varphi'(0)) = (1 - m_2)\|d\|^2$$Simplifying the inequality, we obtain: $$[\overline{\alpha} >] \quad \alpha' > \frac{1 - m_2}{L}$$
This is an exceptional result. It proves that $\alpha'$ is a "large" value (of the same order of magnitude as $1/L$). Therefore, the Gradient Method with Armijo-Wolfe LS or Backtracking converges reliably.

But how fast does it converge? In the quadratic case, it depends on the absence of directions with zero curvature. We need an appropriate generalization of this concept.

Here is the expanded and integrated version of the **Stronger Forms of Convexity** section in English. I’ve woven the mathematical formulas together with the geometric "why" to make it crystal clear.

## Stronger Forms of Convexity: The "Curvature" Engine
To guarantee that the Gradient Method doesn't just wander around aimlessly or take forever to reach the bottom, we need the function to have a **pronounced curvature** toward the minimum. In optimization, we categorize this into three levels of "strictness."
#### Convex Function (The Foundation)
A function $f$ is convex if it always lies **above its tangent plane**.
- **The Formula ($C^1$):**$$f(z) \ge f(x) + \langle\nabla f(x), z - x\rangle \quad \forall x, z$$
- **The Hessian ($C^2$):** $$\nabla^2 f(x) \ge 0 \quad (\text{positive semi-definite})$$
- **The Problem:** A convex function can be **perfectly flat** (like a floor). On a flat surface, the gradient is zero everywhere, meaning the algorithm could stop anywhere, even far from a global minimum.
    
#### Strictly Convex Function (The Unique Minimum)
By using a strict inequality ($>$), we ensure the function has a "bend."
- **Geometric Meaning:** The function is "curved" enough that if a minimum exists, it is **unique**.
- **The Problem:** It can still be **extremely flat** far away. Imagine a very shallow bowl; the gradient might be so small ($10^{-10}$) that the algorithm practically stalls, taking millions of years to reach the center.

#### Strongly Convex Function ($\tau$-Convexity)
It forces the function to grow at least as fast as a **parabola**.
- **The Formula:**$$f(z) \ge f(x) + \langle\nabla f(x), z - x\rangle + \frac{\tau}{2}\|z - x\|^2$$
- **The Hessian Perspective:**$$\nabla^2 f(x) \ge \tau I > 0$$
    This means the **smallest eigenvalue** ($\lambda_{min}$) of the Hessian is at least $\tau$.
    
- **The "Paviment" (Lower Bound):** No matter where you are, the function has a "minimum pendenza" (minimum slope). It cannot flatten out. This "pushes" the gradient to stay large enough to keep the algorithm moving toward the solution.


When a function is both **$L$-smooth** (from your previous notes) and **$\tau$-strongly convex**, we have successfully "trapped" its curvature between two parabolas:$$\tau I \le \nabla^2 f(x) \le LI$$$$\text{meaning: } 0 < \tau \le \lambda_n \le \dots \le \lambda_1 \le L$$
- **$L$ (The Ceiling):** Prevents the function from being too "steep" or "nervous." (Ensures stability).
- **$\tau$ (The Floor):** Prevents the function from being too "flat." (Ensures speed).
#### Efficiency and the "Zig-Zag" Effect
The speed of the Gradient Method is dictated by the **Condition Number** $\kappa$:$$\kappa = \frac{L}{\tau}$$ 
- **If $\kappa \approx 1$:** The function is a **perfect circular bowl**. The gradient points directly to the center. You arrive in very few steps.
- **If $\kappa$ is large (e.g., 1000):** The function is a **long, narrow valley** (like a cigar).
    - The $L$-bound makes you bounce violently between the narrow side-walls.
    - The $\tau$-bound is so small that you barely crawl forward along the long axis.
    - This is the source of the infamous **"Zig-Zagging"** behavior.
    
## Efficiency of the Gradient Method with A-W LS
When the function possesses these "ideal" mathematical characteristics, how does the algorithm behave in practice?
- Efficiency fundamentally behaves as expected based on the worst-case **condition number**: $\kappa = L/\tau$.
- In reality, these properties are strictly necessary only within the basin of attraction $\mathcal{B}(x_*, \delta)$, provided that $\{x^i\} \rightarrow x_*$ (proving this is non-trivial, though it usually happens in practice).
- Using an **Inexact Line Search** (like Armijo-Wolfe) produces a convergence rate $r \approx (1 - \lambda^n/\lambda^1)$ that is slightly "worse" compared to an Exact Line Search, depending on the choice of $m_1$ and $m_2$.
	- **Note:** This slowdown counts overall iterations, not function evaluations ($f$-calls). A smaller $m_1$ or a larger $m_2$ worsens the convergence rate $r$ but makes the search (LS) much faster.
- **The Non-Trivial Trade-off:** A more inexact Line Search reduces the convergence speed per iteration but requires far fewer calculations of function $f$, and it is this saving that is most noticeable in practice.

Fortunately, the default parameter values set in standard solvers work well in almost all operational scenarios.

# References