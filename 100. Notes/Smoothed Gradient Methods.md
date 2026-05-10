---
Data: 2026-05-05T00:41:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Smoothed Gradient Methods
This section introduces a paradigm shift in nonsmooth optimization: rather than using a slow algorithm on a difficult function, we **change the function slightly to make it smooth**, allowing us to apply faster algorithms.

### The Core Idea: "Smoothing" the Function
This approach requires that the nonsmooth convex function $f(x)$ has a specific structure (often called a minimax structure):
$$f(x) = \max \{ x^T A z : z \in Z \}$$
(Note: Non-differentiability occurs because multiple vectors $z$ can be optimal for a single $x$, leading to many different subgradients $A\overline{z} \in \partial f(x)$)

**The Smoothing Technique:** We create a smoothed version, $f_\mu(x)$, by subtracting a strongly convex quadratic penalty weighted by the smoothing parameter $\mu > 0$:
$$f_{\mu}(x) = \max \{ x^{T} A z - \mu ||z||^{2} / 2 : z \in Z \}$$
By adding this penalty, the maximization problem becomes strictly concave, ensuring a unique solution for $z$ for any $x$, which makes $f_\mu \in C^1$ (differentiable).

### Mathematically Speaking: Analysis of Smoothed Gradient
The text provides a complete derivation to show why this approach reaches a complexity of $O(1/\epsilon)$.

**Step 1: Define the Constants and Bounds**
- Let $Z$ be a convex and compact set.
- Define $K = \max \{ ||z||^2 / 2 : z \in Z \}$ as the maximum value of the penalty term over $Z$.
- The relationship between the original function and the smoothed version is:
$$f_\mu(x) \le f(x) \le f_\mu(x) + \mu K$$    
- As $\mu \rightarrow 0$, $f_\mu$ approaches $f$, and the minimum of the smoothed function approaches the true minimum: $\text{argmin} \{ f_\mu(x) \} \rightarrow x_*$.

**Step 2: Lipschitz Smoothness**
- According to Nesterov's theorem, $f_\mu$ is $L$-smooth with the Lipschitz constant:
$$L = \frac{||A||^2}{\mu}$$
- Notice that as $\mu$ becomes smaller (for better accuracy), $L$ grows, meaning the function becomes "less and less Lipschitz" (steeper).
    

**Step 3: Convergence Rate of the Smooth Algorithm**
- For an $L$-smooth function, an accelerated gradient method (ACCG) guarantees an error of:
$$f_\mu(x^i) - f_{\mu,*} \le \frac{2 L D^2}{i^2}$$
    _(Where $D$ is the distance to the optimum and $i$ is the iteration count)_.


**Step 4: Balancing Approximation and Optimization Errors** To ensure an overall error $f(x^i) - f_* \le \epsilon$, we split the error budget:

1. Set the approximation gap to $\epsilon/2$ by choosing $\mu$:    
$$\mu K = \frac{\epsilon}{2} \implies \mu = \frac{\epsilon}{2K}$$
2. Substitute this $\mu$ into the Lipschitz constant formula:
$$L = \frac{||A||^2}{\mu} = \frac{2 ||A||^2 K}{\epsilon}$$
3. The goal is to ensure the optimization error on the smoothed function is also $\le \epsilon/2$:
$$f_{\mu}(x^{i}) - f_{\mu,*} \le \frac{\epsilon}{2}$$

**Step 5: Solving for Iterations ($i$)** Substitute $L$ into the accelerated convergence formula:

$$\frac{2 (\frac{2 ||A||^2 K}{\epsilon}) D^2}{i^2} \le \frac{\epsilon}{2}$$
$$\frac{4 ||A||^2 K D^2}{\epsilon i^2} \le \frac{\epsilon}{2}$$
$$\frac{8 ||A||^2 K D^2}{\epsilon^2} \le i^2$$
$$i \ge \frac{\sqrt{8K} ||A|| D}{\epsilon}$$

**Conclusion:** The number of iterations scales with $1/\epsilon$, yielding **$O(1/\epsilon)$** complexity. This is significantly better than the **$O(1/\epsilon^2)$** required by standard subgradient methods.

### Practical Reality: The "Consistently Slowish" Behavior
Despite the theoretical speedup, smoothed gradient methods often perform poorly in practice.
- **Implementation Difficulty:** One must "pry open the black box" to identify $A$ and $Z$, and estimate $K$ to choose $\mu$, which is a difficult convex maximization problem.
- **The "Long Flat Leg":** Because $L \propto 1/\epsilon$, if the desired error is small, $L$ is huge. Since step sizes are $1/L = O(\mu) = O(\epsilon)$, the algorithm takes microscopic steps at the start.
- **Comparison:** In a doubly-logarithmic chart, subgradient methods drop fast initially but "flatline" at $\epsilon \approx 1e-4$. Smoothed methods eventually reach $\epsilon = 1e-6$, but only after a long "flat" period where no progress seems to occur, often requiring $1e+6$ iterations.
- **Optimization:** The efficiency can be improved by not keeping $\mu$ fixed, but dynamically adjusting it based on information about $f_*$ (e.g., using $\epsilon^i$ that shrinks as the algorithm progresses).

![[Pasted image 20260510154946.png]]

# References