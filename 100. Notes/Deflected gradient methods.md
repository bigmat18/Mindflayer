---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# Deflected Gradient Methods
In the methods we have seen previously, to accelerate convergence we "twisted" the gradient direction by multiplying it by a matrix (e.g., the inverse of the Hessian in Newton's methods, or an approximation $H^i$ in Quasi-Newton methods).

The _twisting_ operation takes the form:

$$d^i = H^i(-\nabla f(x^i))$$

The fundamental problem with this approach is that multiplying a matrix by a vector requires at least **$O(n^2)$** operations per iteration (not even counting the cost of forming or updating the matrix $H^i$). Unless $H^i$ is "very special" (which requires rather dirty tricks), this cost is prohibitive for large-scale problems with millions of variables.

## The Cheaper Alternative: Deflecting
To drastically reduce computational costs, we introduce a much cheaper alternative: _deflecting_. Instead of rotating the gradient vector by multiplying it by a matrix, we simply **add** another vector $v^i$ to it:

$$d^i = -\nabla f(x^i) + v^i$$

_Explanation:_ The addition of two vectors strictly requires **$O(n)$** operations. The computational savings are immense. The real problem, however, is: how do we choose this vector $v^i$ within the entire infinite space of $\mathbb{R}^n$ while keeping the computations cheap?

## The Core Idea: Reusing History
The simplest and most brilliant idea is to define $v^i$ by simply scaling the direction used in the previous iteration by a scalar parameter $\beta^i$:

$$v^i = \beta^i d^{i-1}$$

Our new direction update rule thus becomes:

$$d^i = -\nabla f(x^i) + \beta^i d^{i-1}$$

What happens if we apply this formula iteratively? Let's assume we start at the first step with the classic negative gradient ($v^0 = 0$). By unrolling the recursion, we discover that the direction at step $i$ takes this form:

$$d^i = -\left[ \sum_{h=1}^i \gamma^h \nabla f(x^h) \right]$$

_(for some sequence of weights $\gamma^h$ depending on the various $\beta$)_.

_Explanation:_ This expression reveals that the current descent direction is nothing more than the opposite of an **aggregated sum of all past gradients**. Without having to store giant matrices as the BFGS method does, we are implicitly incorporating the "history" of the computation into a single vector, achieving a similar effect but at a fractional cost ($O(n)$).

## The Problem of Guaranteeing Descent

Having a computationally cheap idea is great, but the algorithm must actually work. We need to guarantee that the newly found direction is effectively a descent direction, meaning the initial slope must be negative: $\varphi_{x^i,d^i}'(0) < 0$.

- In the case of _twisting_ (Quasi-Newton), guaranteeing descent was "easy": we just had to ensure mathematically that the matrix used was positive definite ($H^i \ge 0$).
    
- In the case of _deflecting_, choosing a scalar parameter $\beta^i$ that consistently guarantees descent is a **nontrivial** problem.
    

We might think of an obvious solution: if we let $\beta^i \to 0$, the "history" vector vanishes and we get back the standard gradient direction ($d^i \to -\nabla f(x^i)$), which we know for sure points downhill. But if we zero out the weight of the history, we are back to square one: we lose the benefits of the method and fall back into the well-known **slowness** of the pure gradient method.

Therefore, we need better mathematical ideas to calculate this $\beta^i$ optimally. This necessity lays the foundation for the famous **Nonlinear Conjugate Gradient methods**, where exact formulas are derived to compute the deflection parameter $\beta^i$ in a way that mimics the properties of second-order methods.
# References