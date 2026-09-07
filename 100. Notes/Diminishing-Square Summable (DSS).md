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
# Diminishing-Square Summable (DSS)

To prove the algorithm converges, we must ensure the function's slope doesn't grow to infinity.
- A **Lipschitz condition** is assumed: $||g^i|| \le L$.
- If the function is not globally $L$-Lipschitz, the proof still holds as long as we can guarantee that the iterates remain within a bounded region of space ($||x^i|| \le M < \infty$).

We want to prove that the algorithm with the **DSS stepsize converges to the minimum $f_*$**. We proceed by contradiction: **suppose the algorithm fails and never approaches the solution.**

We assume there exists an error margin $\delta > 0$ such that, at every iteration $i$, the function always remains distant from the optimum:
$$f(x^i) - f_* \ge \delta/2 > 0$$
We start from the fundamental relationship of the squared distance from the optimum. We substitute our contradiction hypothesis ($f_* - f(x^i) \le -\delta/2$) and the subgradient upper bound ($||g^i||^2 \le L^2$):
$$||x^{i+1} - x_*||^2 \le ||x^i - x_*||^2 + 2\alpha^i(f_* - f(x_i)) + (\alpha^i)^2||g^i||^2$$
$$||x^{i+1} - x_*||^2 \le ||x^i - x_*||^2 - \delta\alpha^i + L^2(\alpha^i)^2$$
This inequality **describes the change over a single step**. To evaluate the cumulative effect after $k$ iterations, we sum the contributions by induction:
$$||x^{k+1} - x_*||^2 \le ||x^1 - x_*||^2 + v^k$$
Here, $v^k$ represents the accumulation of all steps and errors generated up to that point:
$$v^k = \underbrace{-\delta\sum_{i=1}^{k}\alpha^i}_{\text{Progress towards optimum (< 0)}} + \underbrace{L^2\sum_{i=1}^{k}(\alpha^i)^2}_{\text{Accumulated error (> 0)}}$$
##### The Core Logic: The Power of the DSS Rule
The Diminishing-Square Summable (DSS) rule imposes two precise mathematical constraints (e.g., $\alpha^i = 1/i$):
1. **The sum of the stepsizes diverges:** $\sum_{i=1}^{\infty}\alpha^i = \infty$.
2. **The sum of the squared stepsizes converges to a finite value:** $\sum_{i=1}^{\infty}(\alpha^i)^2 < \infty$.

Because of these two properties, as $k \rightarrow \infty$, the "progress" term (which diverges to negative infinity) mathematically dominates the "error" term (which caps at a maximum limit).
- **Result:** $v^k \rightarrow -\infty$.
    
There will inevitably be an iteration $k$ where the negative value of $v^k$ exceeds the absolute magnitude of the initial distance $||x^1 - x_*||^2$. At that exact moment, a mathematical short-circuit occurs:
$$0 \le ||x^{k+1} - x_*||^2 \le ||x^1 - x_*||^2 + v^k < 0$$
Since a squared distance cannot be less than zero, our initial assumption ("the algorithm never approaches the optimum") is blatantly **false**. The algorithm inexorably converges.

##### Practical Reality
Despite the elegant mathematical guarantee, the DSS rule is highly discouraged for practical use for two critical reasons:
- **Lack of Monotonicity:** The proof certifies that there will be an iterate $x^i$ arbitrarily close to $x_*$. However, it does not guarantee that the algorithm stays there: the next iterate $x^{i+1}$ could "bounce" very far away.
- **Blind Stepsize:** A preset stepsize $\alpha$ that turns out to be "good" at iteration $i$ can be "very bad" at the next iteration. The method has no local control over what happens at every single step; it blindly relies solely on a "long-term average" guarantee.

# References