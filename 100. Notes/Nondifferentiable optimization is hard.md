---
Data: 2026-05-05T00:24:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
Area: "[[Master's degree]]"
---
# Nondifferentiable Optimization is Hard

The core thesis of this section is that dealing with nondifferentiable (nonsmooth) convex functions is not just a minor technical inconvenience; it makes the optimization process **orders of magnitude slower** and breaks many of the standard tools and assumptions we rely on in classic calculus.

Here is a detailed breakdown of why this happens, categorized by complexity, algorithmic behavior, and mathematical properties.

### The Mathematical Speed Limit (Complexity Analysis)
The text compares the theoretical convergence rates (how many iterations are required to reach an error of at most $\epsilon$) for smooth functions ($f \in C^1$) versus nonsmooth functions ($f \notin C^1$). The notation $O(\cdot)$ represents an upper bound (how fast we can guarantee convergence), while $\Omega(\cdot)$ represents a lower bound (the fundamental speed limit; you cannot mathematically do better than this).

![[Pasted image 20260509151548.png]]

**For Strongly Convex (T-convex) Functions:**
- **Smooth ($C^1$) + L-smooth:** The error decreases exponentially fast. It requires only **$O(\log(1/\epsilon))$** iterations.
- **Nonsmooth ($\notin C^1$) + L-Lipschitz:** The speed limit drops drastically. It requires at least **$\Omega(L^2/\epsilon)$** iterations.

**For General Convex Functions:**
- **Smooth ($C^1$) + L-smooth:** Convergence is slower than the strongly convex case, requiring **$O(1/\sqrt{\epsilon})$** iterations.
- **Nonsmooth ($\notin C^1$) + L-Lipschitz:** This is the worst-case scenario. The algorithm will require at least **$\Omega(L/\epsilon^2)$** iterations.

**Conclusion:** Moving from a smooth function to a nonsmooth function changes the convergence rate from a fast logarithmic or inverse-square-root curve to a painfully slow inverse-square curve ($\epsilon^2$ in the denominator means if you want a high precision, the number of iterations explodes).


### The Failure of the "Fixed Step Size"
In classic smooth optimization, you can often use a constant step size (learning rate) $\alpha > 0$ and the algorithm will naturally slow down and settle at the minimum. The text mathematically proves that **a fixed step size "cannot work" for nonsmooth functions**.

**The Infinite Oscillation Example:** Consider the simple nonsmooth function $f(x) = L|x|$. The true minimum is at $x_* = 0$, where $f_* = 0$. Let's trace a gradient descent algorithm with a fixed step size $\alpha$

![[Pasted image 20260509151715.png|300]]

1. **Initialization:** We start at a point slightly to the left of the minimum: $x^0 = -\alpha L/2$.
![[Pasted image 20260509151734.png | 300]]

2. **Iteration 1:**
    - Since we are on the negative side of the absolute value, the subgradient is $g^1 = -L$.
	![[Pasted image 20260509151755.png| 300]]
	
	- We update the position: $x^1 = x^0 - \alpha g^1$.
	![[Pasted image 20260509151829.png | 300]]

    - Plugging in the numbers: $x^1 = (-\alpha L/2) - \alpha(-L) = (-\alpha L/2) + \alpha L = \alpha L/2$.
    ![[Pasted image 20260509151930.png | 300]]
    
    - _Result:_ The algorithm jumped entirely over the minimum to the positive side.
    
3. **Iteration 2:**
    - Now we are on the positive side, so the subgradient is $g^2 = L$.
	![[Pasted image 20260509151947.png | 300]]
	
    - We update the position: $x^2 = x^1 - \alpha g^2$.
    - Plugging in the numbers: $x^2 = (\alpha L/2) - \alpha(L) = -\alpha L/2$.
    - _Result:_ $x^2 = x^0$. We are exactly back where we started.
    
 4. **Iteration 3**:  $g^3 = -L, x^1 = x^0 - \alpha g^1$
	![[Pasted image 20260509152125.png | 300]]

	![[Pasted image 20260509152148.png | 300]]

**The Consequence:** The algorithm will cycle infinitely between $-\alpha L/2$ and $\alpha L/2$. The best function value it ever achieves is at these points, which is $f_{best} = L |\pm\alpha L/2| = L^2\alpha/2$. Therefore, the gap between the best found value and the true minimum is __$f_{best} - f = L^2\alpha/2$

If we choose a standard step size like $\alpha = 1/L$, the error gets stuck at $O(L)$, and the algorithm never reaches the minimum. To actually converge, a nonsmooth algorithm _must_ force the step size to shrink over time ($\alpha^i \rightarrow 0$).

### Why Nonsmooth Optimization is Fundamentally Harder
The section concludes by summarizing the stark differences between the smooth and nonsmooth worlds.
#### A. Descent Guarantees
- **Smooth ($C^1$):** The negative gradient $d = -\nabla f(x)$ is unique. Moving in this direction _guarantees_ a decrease in the function value: $f(x + \alpha d) < f(x)$ for any sufficiently small $\alpha \ge 0$.
- **Nonsmooth ($\notin C^1$):** There can be many different subgradients, and the algorithm calculates the direction based on an arbitrary one provided by the "oracle": $d = -[g \in \partial f(x)]$. Because you cannot choose which subgradient you get, moving along $d$ might actually _increase_ the function value, meaning $f(x + \alpha d) \ge f(x)$ for all $\alpha$.
    
#### B. The Proxy for Optimality (Stopping Criteria)
How do we know when to stop the algorithm?

- **Smooth ($C^1$) - A Two-Sided Proxy:** The norm of the gradient, $||d||$, is a perfect two-way indicator.
    - If $||d||$ is small $\iff$ $f(x)$ is close to the minimum $f_*$.
    - Therefore, checking if $||d|| \le \epsilon$ is a highly effective and reliable stopping criterion. Because the gradient naturally shrinks to zero, a fixed step size works fine because the update step $||x^{i+1} - x^i|| \rightarrow 0$ automatically.
        
- **Nonsmooth ($\notin C^1$) - A One-Sided Proxy:** The norm of the subgradient $||d||$ is a broken indicator.
    - It is true that if $||d||$ is small, then $f(x)$ is close to $f_*$.
    - **HOWEVER**, the reverse is false: $f(x)$ being close to $f_*$ does **not** imply that $||d||$ is small. Even if you are standing exactly on the optimal point $x = x_*$, the oracle might hand you a subgradient with a massive norm.
    - Therefore, the condition $||d|| \le \epsilon$ is a completely ineffective stopping criterion because it almost never actually happens in practice.
    - Because the subgradient doesn't shrink to zero as you approach the minimum, the only way to ensure the algorithm eventually stops moving ($||x^{i+1} - x^i|| \rightarrow 0$) is to manually force the step size $\alpha^i$ to approach zero—but not too fast, or the algorithm will freeze before reaching the minimum.
# References