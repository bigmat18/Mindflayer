---
Data: 2026-05-05T00:39:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Subgradient Methods

This section dives into the algorithmic construction of **Subgradient Methods**, which are the fundamental tools used to minimize nondifferentiable convex functions. Since we cannot guarantee that moving opposite to a subgradient strictly decreases the function value, we must rely on a different geometric property to ensure progress.

### 1. The Fundamental Relationship: Getting Closer to the Optimum

As established previously, any negative subgradient ($-g$) at a point $x$ points "towards" the global minimum $x_*$. This means that taking an appropriate step along $-g$ will bring the next iterate $x^{i+1}$ physically closer to $x_*$.

The algorithm's update rule is defined as:

$$x^{i+1} = x^i - \alpha^i g^i$$

_(where $\alpha^i$ is the stepsize and $g^i \in \partial f(x^i)$ is the subgradient)._

To prove this works, we analyze the squared distance to the optimum, $||x^{i+1} - x_*||^2$. We expand this using the update rule:

$$||x^{i+1} - x_*||^2 = ||x^i - \alpha^i g^i - x_*||^2$$

$$= ||x^i - x_*||^2 + 2\alpha^i \langle g^i, x_* - x^i \rangle + (\alpha^i)^2 ||g^i||^2$$

Using the subgradient inequality, we know that $\langle g^i, x_* - x^i \rangle \le f(x_*) - f(x^i)$. Substituting this in, we get the **Fundamental Relationship**:

$$||x^{i+1} - x_*||^2 \le ||x^i - x_*||^2 + \underbrace{2\alpha^i (f_* - f(x^i))}_{\text{Negative Term (< 0)}} + \underbrace{(\alpha^i)^2 ||g^i||^2}_{\text{Positive Term (> 0)}}$$

**Why this matters:**

- The **negative term** pulls the distance down because $f_* \le f(x^i)$.
    
- The **positive term** pushes the distance up.
    
- Because the negative term depends linearly on $\alpha^i$ and the positive term depends quadratically on $(\alpha^i)^2$, for a sufficiently small (short) step $\alpha > 0$, the linear negative term will dominate the quadratic positive term.
    
- **Result:** $||x^{i+1} - x_*||^2 < ||x^i - x_*||^2$, meaning the new point $x^{i+1}$ is strictly closer to $x_*$ than $x^i$ was.
    

---

### 2. Stepsize Strategy 1: Diminishing-Square Summable (DSS)

If we know nothing about the function, how do we guarantee convergence? We must use a stepsize that shrinks over time, but not too fast. This is the Diminishing-Square Summable (DSS) rule.

**The DSS Conditions:**

1. $\sum_{i=1}^\infty \alpha^i = \infty$ (The sum of the steps goes to infinity, allowing the algorithm to travel any required distance).
    
2. $\sum_{i=1}^\infty (\alpha^i)^2 < \infty$ (The sum of the squared steps is finite, ensuring the error term eventually vanishes).
    

A classic example is $\alpha^i = 1/i$. The sequence $\alpha^i \searrow 0$, but not so fast that the series converges.

**Convergence Analysis of DSS:** Assuming the subgradients are bounded ($||g^i|| \le L$, meaning the function is Lipschitz continuous), DSS mathematically guarantees that for any small $\epsilon > 0$, there exists an iteration $i$ where $f^i - f_* \le \epsilon$. **The Catch:** While incredibly robust (it works even without evaluating $f(x^i)$), the practical convergence speed of DSS is abysmal. The sequence of function values is not monotonic, meaning a "good" step at iteration $i$ can be followed by a "very bad" step at $i+1$, as we only have control over the "long-term average" of the step sizes.

---

### 3. Stepsize Strategy 2: Polyak Stepsize (The Ideal Scenario)

Let's look back at the fundamental inequality. If we treat it as a quadratic function of the stepsize $\alpha$, $\phi(\alpha) = a\alpha^2 + b\alpha$, where $a = ||g^i||^2 > 0$ and $b = 2(f_* - f^i) < 0$, we can find the exact stepsize that minimizes the distance to the optimum.

The minimum of this parabola occurs at $\alpha_* = -b/(2a)$. Plugging in our terms gives the **Polyak Stepsize**:

$$\alpha_*^i = \frac{f^i - f_*}{||g^i||^2}$$

Any stepsize in the range $\alpha^i \in (0, 2\alpha_*^i)$ guarantees a strict decrease in distance: $||x^{i+1} - x_*||^2 < ||x^i - x_*||^2$.

**Efficiency of the Polyak Stepsize:** Using the optimal step $\alpha_*^i$ implies an **$O(1/\epsilon^2)$** complexity, which is vastly better in practice. **The fatal flaw:** The Polyak formula requires knowing the exact optimal value $f_*$, which we almost never know in real-world problems.

---

### 4. Stepsize Strategy 3: Target Level Stepsize

"If you don't know $f_*$, estimate it, but be ready to revise your estimate". Since the Polyak stepsize requires $f_*$, we can replace it with a dynamically updated "Target Level" approximation.

Here is the pseudocode detailing how this logic is implemented in practice:

```c
procedure x = SGPTL(f, x, i_max, \beta, \delta_0, R, \rho)
    r = 0; 
    \delta = \delta_0; 
    f_ref = \overline{f} = f(x); 
    i = 1;

    while (i < i_max) do
        g \in \partial f(x); 
        
        // Calculate stepsize using Target Level instead of f_*
        \alpha = \beta * (f(x) - (f_ref - \delta)) / ||g||^2;
        
        // Update position
        x = x - \alpha * g; 
        
        // Check for "Good improvement"
        if (f(x) \le f_ref - \delta/2) then {
            f_ref = \overline{f}; 
            r = 0; 
        }
        // Check for "Too many steps without improvement"
        else if (r > R) then {
            \delta = \delta * \rho; 
            r = 0; 
        }
        // Accumulate distance without improvement
        else {
            r = r + \alpha * ||g||;
        }
        
        // Update record value and increment
        \overline{f} = \min\{\overline{f}, f(x)\}; 
        i = i + 1;
```

(Note: The source transcript contains a typo "f(f(x)..." which represents the `if` statement for checking the condition)

**Understanding the Pseudocode:**

- **The Target:** The algorithm uses a reference value $f_{ref}$ (the best value seen recently) and a threshold $\delta$. The target level is $f_{ref} - \delta \approx f_*$.
    
- **The Stepsize:** It computes $\alpha$ exactly like the Polyak stepsize, but swaps $f_*$ for the target level $(f_{ref} - \delta)$.
    
- **Updating Success:** If the algorithm reaches or exceeds half of the targeted gap ($f(x) \le f_{ref} - \delta/2$), the guess was good. It updates the reference value to the new lowest record ($\overline{f}$) and resets the patience counter ($r=0$).
    
- **Updating Failure:** The variable $r$ acts as a tracker for how much distance the algorithm has traveled without seeing a significant improvement ($r = r + \alpha ||g||$). If it travels too far ($r > R$), it assumes the target was too ambitious. It shrinks the threshold $\delta$ by a factor of $\rho \in (0,1)$ and resets the counter.
    

While this creates a working algorithm that converges ($\overline{f}^i \rightarrow f_*$), it introduces many ugly hyper-parameters ($\rho \in (0,1)$, $\beta \in (0,2)$, initial $\delta_0 > 0$, patience limit $R > 0$). Furthermore, it still lacks a reasonable stopping criterion other than "stop after a while".

---

### 5. Deflected Subgradient Methods

"Want a better direction? Use a better model!" Since moving strictly along the negative subgradient causes erratic zig-zagging, we can "deflect" the direction using momentum from previous steps (similar to conjugate gradient methods for smooth functions).

The new direction $d^i$ is a combination of the current subgradient and the previous direction:

$$d^i = \gamma^i g^i + (1 - \gamma^i) d^{i-1}$$

$$x^{i+1} = x^i - \alpha^i d^i$$

To maintain theoretical convergence, strict and sometimes "funny" rules are needed for the mixing parameter $\gamma^i$ and stepsize $\alpha^i$:

- **Stepsize-restricted (Polyak approach):** $\alpha^i = \beta^i (f^i - f_*) / ||d^i||^2$ with the restriction $\beta^i \le \gamma^i$. As deflection increases, the stepsize must decrease.
    
- **Deflection-restricted (DSS approach):** A complex formula restricts $\gamma^i$ based on the previous step size and error: $\frac{\alpha^{i-1}||d^{i-1}||^2}{(f^i - f_*) + \alpha^{i-1}||d^{i-1}||^2} \le \gamma^i$. As $f(x^i) \rightarrow f_*$, the allowed deflection goes down.
    

Alternatively, $\gamma^i$ can be found via a closed-formula projection: $\gamma^i \in \text{argmin} \{ ||\gamma g^i + (1-\gamma) d^{i-1}||^2 : \gamma \in [0,1] \}$. While deflection does help stabilize the path in practice, the improvement is incremental and "not much" in the grand scheme of nonsmooth optimization limits.
# References