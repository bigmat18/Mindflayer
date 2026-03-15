---
Data: 2026-03-04T15:29:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Unconstrained Multivariate Optimality and Convexity]]"
Area: "[[Master's degree]]"
---
# Unconstrained Multivariate Optimization

## The Global Optimization Problem
In multivariate optimization, we extend our objective function to operate on an $n$-dimensional space. Mathematically, this is defined as:
$$f:\mathbb{R}^{n}\rightarrow\mathbb{R}$$
This means the function evaluates a vector of variables $f(x_{1},x_{2},...,x_{n}) = f(x)$ to produce a single scalar value. To establish any rigorous mathematical guarantees for finding an optimum, we require the function $f$ to be [[Optimization Difficult#Lipschitz Continuity|Lipschitz Continuity]], often denoted simply as $f$ being $L$-c.

The core issue with unconstrained global optimization is the **curse of dimensionality**. There is very bad news on the theoretical front: no algorithm can find a global minimum in fewer than $\Omega((LD/\epsilon)^{n})$ operations. Because the dimension $n$ acts as an exponent, the computational effort explodes, making the problem not really doable unless $n=3/5/10$ tops. 

While exact global optimization is generally intractable in high dimensions, several practical approaches exist depending on the nature of the function:
- **Grid Search:** We can approach the problem in $O((LD/\epsilon)^{n})$ using a multidimensional grid with a small enough step. This is the standard approach to hyperparameter optimization, though the exact diameter $D$ and Lipschitz constant $L$ are usually unknown.
- **Analytic Functions:** If $f$ is analytic, clever spatial Branch & Bound (B&B) techniques can systematically explore the space to yield the provable global optimum.
- **Black-Box Functions:** When no derivatives are available (typically the case for complex systems), many effective heuristics can provide good, albeit not provably optimal, solutions.

In all these practical scenarios, the computational complexity grows "fast" as $n$ grows. Finding good global solutions is hard in practice, and proving their optimality is even worse. 

The only major exception is **if $f$ is convex**; in that specific case, the math simplifies beautifully because every local minimum is also a global minimum (global = local).

## The Shift to Local Optimization
Given the exponential difficulty of global searches, **local optimization is much better** and computationally tractable. 

The mathematical results for local optimization are generally, and surprisingly, analogous to the multivariate quadratic case. Most convergence results are **dimension-independent**. This means the number of iterations required to find a minimum does not explicitly depend on $n$, or at least not exponentially. This efficiency is not completely surprising, as building linear and quadratic models of the function is a staple of these algorithms.

However, escaping the curse of dimensionality does not mean all local algorithms are fast. There are several caveats to consider:
- **Convergence Speed:** The actual speed of convergence may be rather low, sometimes characterized as "badly linear" or worse.
- **Computational Cost:** The cost of computing $f$ and its derivatives necessarily increases with $n$. For modern large-scale problems where $n \approx 10^{9}$, even an algorithm with a polynomial complexity like $O(n^{2})$ requires too much memory and time.
- **Hidden Constants:** Some dependency on the dimension $n$ may be hidden mathematically within the constants of the $O(\cdot)$ notation.

Ultimately, large-scale local optimization is highly doable if you have access to the function's derivatives. The next mathematical hurdle is that defining and computing derivatives in $\mathbb{R}^{n}$ is significantly more complex than in standard univariate calculus.
# References