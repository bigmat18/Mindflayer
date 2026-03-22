---
Data: 2026-03-22T20:02:00
Tags:
  - note
  - youngling
Connection:
Area:
---
# Twisted Gradient Methods

## $\infty$-ly many possible directions

Up to this point, we have relied on one outstanding assumption: the descent direction is always the exact negative gradient, meaning $d^i = -\nabla f(x^i)$. But is this strict choice really needed?

To answer this, we must analyze the two crucial arguments that guarantee the convergence of the method:

1. **$\varphi_i'(0) = -||\nabla f(x^i)||^2$**: This means that "far from $X_*$ the derivative is very negative". Far from the optimum, the initial slope along the gradient direction is strongly negative, guaranteeing that the function will decrease.
    
2. **"you can get a non-vanishing fraction of the descent promised by $\varphi_i'(0)$"**. This means the algorithm successfully "cashes in" a significant portion of that promised descent.
    

Point 2 is guaranteed by the step-size strategies we have seen so far ("exact" LS, Armijo, or Fixed Stepsize combined with L-smoothness). These techniques ensure that the stepsize $\alpha_i$ does not $\to 0$ "too fast". Consequently, we achieve a "significant decrease at each step unless $||\nabla f(x^i)|| \to 0$".

The mathematical revelation is that **point 2 does not really depend on the chosen direction**. There are many other directions that ensure point 1 holds (within some scaling factor).

To demonstrate this, the slides propose a parodied algorithm: the _twisted gradient algorithm_. Imagine taking the standard direction $-\nabla f(x^i)$ and rotating it by a fixed angle, for example, $\pi/4$ (45 degrees). We can write this mathematically using a rotation matrix $R$:

$$d^i = R(-\nabla f(x^i))$$

Calculating the initial directional derivative with this new direction, we get:

$$\varphi_i'(0) = -||\nabla f(x^i)||^2 \cos(\pi/4) < 0 \quad \text{(check)}$$

_Explanation:_ Because the cosine of 45 degrees is a positive number (about 0.707), the final result remains strictly negative. The direction still points "downhill". Consequently, the convergence proofs we have seen so far carry forward largely unchanged.

We are not limited to just $\pi/4$: the angle $\theta$ just needs to be not too close to $\pi/2$ (90 degrees), which would make $\cos(\theta)$ "not too small" (tending to zero). This means there are $\infty$-ly many feasible angles $\theta$, and for each one, there are $\infty$-ly many directions $d \ne -\nabla f$ and $\infty$-ly many rotation matrices $R$ that produce a valid direction.

## Convergence of general descent methods

We can generalize this concept by rigorously defining what constitutes a "descent direction":

$$\frac{\partial f}{\partial d^i}(x^i) < 0 \equiv \langle d^i, \nabla f(x^i) \rangle < 0 \equiv \cos(\theta^i) < 0$$

_Explanation:_ A direction is a descent direction if and only if the directional derivative is negative. This is equivalent to saying the dot product between the direction and the gradient is negative, which in turn means the angle $\theta^i$ between the two vectors has a negative cosine (the angle is greater than 90 degrees relative to the gradient, thus pointing in the opposite half-space).

Intuitively, this means "$d^i$ points roughly in the same direction as $-\nabla f(x^i)$". In a multi-dimensional space, there is a whole half space of descent directions, offering a lot of flexibility.

To prove global convergence for any of these directions, we rely on **Zoutendijk's Theorem**: If $f \in C^1$, $f$ is L-smooth, the minimum is bounded below ($f_* > -\infty$), and we use a Line Search satisfying Armijo and Wolfe $(A) \cap (W)$, then:

$$\sum_{i=1}^\infty \cos^2(\theta^i) ||\nabla f(x^i)||^2 < \infty$$

**Consequence of the theorem:** If we guarantee that $\sum_{i=1}^\infty \cos^2(\theta^i) = \infty$ (which means our directions $d^i$ do not get perpendicular, or $\perp$, to $\nabla f(x^i)$ "too fast"), the only mathematical way for Zoutendijk's sum to remain finite is if the sequence of the gradients tends to zero: $\{||\nabla f(x^i)||\} \to 0$. This proves convergence.

The simple case occurs when we bound the angle so it stays somewhat away from 90 degrees: $\cos(\theta^i) \ge \overline{\theta} > 0$ (bounded away from 0). The standard gradient method we have studied so far is just the obvious case where the angle is 0, and therefore $\cos^2(\theta^i) = 1$.

**Conclusion:** We have established that there are very many directions $d^i$ to choose from, but which $d^i$ is actually better than $-\nabla f$? It is not clear if you only look to the first-order model. To find out, we have to look farther, introducing more complex mathematical models (like the second-order models used in Newton-type methods).


# References