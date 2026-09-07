---
Data: 2026-05-10T15:19:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Polyak Stepsize

As previously established, the practical convergence speed of the [[Diminishing-Square Summable (DSS)]] rule is abysmal. Using a preset, "blind" stepsize means we have no local control over our trajectory.

To find a better way, we must look again at the fundamental distance inequality:
$$||x^{i+1}-x_*||^2 \le ||x^i-x_*||^2 + 2\alpha^i(f_*-f^i) + (\alpha^i)^2||g^i||^2$$

If we actually knew the exact minimum value of the function ($f_*$), we could mathematically calculate the absolute best stepsize $\alpha^i$ to take at any given moment. Polyak's approach asks: _what if we pretend we know it?_

##### Mathematical Derivation: Finding the "Perfect" Step
Let's isolate the part of the inequality that depends on the stepsize $\alpha^i$. We can treat this as a simple 1D quadratic function (a parabola) representing the change in distance:
$$\phi(\alpha) = a\alpha^2 + b\alpha$$
Where the coefficients are:
- $a = ||g^i||^2$ (This is strictly positive, meaning the parabola opens upwards).
- $b = 2(f_*-f^i)$ (Since $f_*$ is the global minimum, this term is strictly negative).

Because $b < 0$, we know that for a certain range of positive stepsizes, this parabola drops below zero:
$$b < 0 \Rightarrow \phi(\alpha) < 0 \forall \alpha \in (0, 2\alpha_*)$$
**This is massive:** it means if we pick an $\alpha$ in this range, the change in distance is strictly negative, guaranteeing we get closer to the optimum!

The absolute lowest point of this parabola (the stepsize that shrinks the distance the most) is found at the vertex $\alpha_* = -b/2a$. Substituting our values for $a$ and $b$, we get the **Polyak Stepsize (PSS)** formula:
$$\alpha_*^i = (f^i-f_*)/||g^i||^2 [\ge 0]$$

By using this exact stepsize (or any $\alpha^i \in (0, 2\alpha_*^i)$), we force the algorithm to be strictly monotonic with respect to the optimal solution:
$$||x^{i+1}-x_*||^2 < ||x^i-x_*||^2$$

##### Efficiency Analysis: Proving the Convergence Rate
Now, let's see how fast this optimal stepsize actually converges.
- First, we assume the **subgradients are bounded:** $||g^i|| \le L$.

Because PSS guarantees we strictly get closer to the optimum, our current distance is always bounded by our starting distance:
$$(PSS) \Rightarrow ||x^{i+1}-x_*|| < ||x^i-x_*|| \Rightarrow ||x^i-x_*|| < ||x^1-x_*|| < \infty \forall i$$
If we plug the optimal stepsize $\alpha^i = \alpha_i^*$ back into our original distance inequality and rearrange the terms, we get a measure of how much progress we make per step:
$$(f^i-f_*)^2/||g^i||^2 \le ||x^i-x_*||^2 - ||x^{i+1}-x_*||^2$$
Because subgradient methods fluctuate, we track the "record" best value found up to iteration $i$:
$$\overline{f}^i = min\{f^h : h \le i\}$$
By substituting the record value $\overline{f}^i$ and the upper bound $L^2$ into our progress equation, we get a worst-case bound:
$$\Rightarrow \frac{(\overline{f}^i-f_*)^2}{L^2} \le \frac{(f(x^i)-f_*)^2}{||g^i||^2} \le ||x^i-x_*||^2 - ||x^{i+1}-x_*||^2$$
If we sum this inequality from the first iteration up to iteration $k$, the intermediate distance terms cancel each other out (a telescoping sum), leaving only the initial and final distances:
$$k\frac{(\overline{f}^k-f_*)^2}{L^2} \le ||x^1-x_*||^2 - ||x^{k+1}-x_*||^2 \le ||x^1-x_*||^2$$
Rearranging to solve for the optimality gap ($\overline{f}^k-f_*$), we arrive at the final convergence rate:
$$\overline{f}^k-f_* \le L||x^1-x_*||/\sqrt{k} \Rightarrow O(1/\epsilon^2)$$

##### The Harsh Practical Reality
While Polyak's stepsize provides a solid theoretical foundation, it hits two massive walls in practice:
1. **The Oracle Paradox:** The formula requires $f_*$. In real-world optimization problems, you don't know the minimum value of the function beforehand—that is exactly what you are writing an algorithm to find! (The "Good news: Polyak would be optimal if we knew $f_*$, which we don't" is a bit of mathematical irony).
2. **The $O(1/\epsilon^2)$ Bottleneck:** Even with this mathematically "perfect" stepsize, the algorithm is vastly better than DSS, but still objectively slow. The $O(1/\epsilon^2)$ rate means that to reach a moderate accuracy of $\epsilon$ = 1e-3, it requires **1e+6 iterations**. Achieving high precision ($< \text{1e-4}$) is computationally impractical.

# References