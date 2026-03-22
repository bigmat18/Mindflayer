---
Data: 2026-03-21T12:53:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Unconstrained Multivariate Optimality and Convexity]]"
Area: "[[Master's degree]]"
---
# Unconstrained Multivariate Optimization
When moving from one-dimensional to multivariate optimization, the objective function is defined as $f: \mathbb{R}^n \to \mathbb{R}$, which can be written as $f(x_1, x_2, \dots, x_n) = f(x)$. While the goal remains finding the minimum value, the "geometry" of $\mathbb{R}^n$ introduces significant theoretical and computational hurdles.

## Unconstrained Global Optimization
Finding the global minimum—the absolute lowest point—in multiple dimensions is an incredibly daunting task. To ensure progress, it is necessary for $f$ to be $L$-continuous ([[Optimization Difficult#Lipschitz Continuity|Lipschitz continuous]]).

#### The Problem: The Curse of Dimensionality
There is a fundamental theoretical limit to how fast an algorithm can find a global optimum:
- **Lower Bound Complexity**: No algorithm can work in less than $\Omega((LD/\epsilon)^n)$.
- **Exponential Growth**: The computational effort required grows exponentially with the number of variables $n$.
- **Curse of Dimensionality**: Global optimization is generally not doable unless $n$ is very small, typically $n = 3, 5,$ or $10$ at most.

#### Global Search Strategies
Despite these difficulties, several practical approaches exist to hunt for the global minimum:
- **Multidimensional Grid Search**: It is possible to achieve the optimum in $O((LD/\epsilon)^n)$ using a grid with a small enough step size. This is the standard approach for hyperparameter optimization, though the constants $D$ (diameter) and $L$ (Lipschitz constant) are often unknown.
- **Analytic Functions**: If the analytical form of $f$ is known, clever spatial Branch & Bound (B&B) methods can find the global optimum.
- **Black-box Heuristics**: When $f$ is a "black box" and derivatives are unavailable, many effective heuristics can provide good solutions, though they are not provably optimal.

In summary: **Finding good global solutions is hard in practice, and proving optimality is even worse unless the function $f$ is convex**. If the function is **convex**, then every local minimum is also a global minimum.

## Unconstrained Local Optimization
Since finding a global minimum is often intractable, we usually settle for local optimization. In this context, the computational outlook is much better.

### Dimension Independence
Unlike the global case, local optimization is significantly more efficient and scales better:
- **Analogous to Quadratic Case**: Results are generally surprisingly similar to the multivariate quadratic case.
- **Dimension-Independent Complexity**: Most convergence results do not explicitly depend on $n$. If there is a dependency, it is typically not exponential.
- **Model-Based**: This efficiency stems from the fact that linear and quadratic models are staples of local optimization.

### Computational Reality and Limits
A "dimension-independent" theory does not necessarily mean the algorithm is instantaneous in practice:
- **Convergence Speed**: The speed may still be low, sometimes characterized as "badly linear" or worse.
- **Iteration Cost**: The cost of computing $f$ and its derivatives necessarily increases as $n$ grows.
- **Large-Scale Challenges**: For extremely large problems ($n \approx 10^9$), even an $O(n^2)$ complexity is too high for practical computation.
- **Hidden Constants**: Some dependency on the dimension $n$ might be hidden within the $O(\cdot)$ constants of the algorithm.

Despite these hurdles, **large-scale local optimization is doable if derivatives are available**. However, derivatives in $\mathbb{R}^n$ (Gradients, Jacobians, and Hessians) are significantly more complex than those in $\mathbb{R}$.

# References