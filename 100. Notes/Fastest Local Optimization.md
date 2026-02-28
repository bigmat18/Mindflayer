---
Data: 2026-02-27T13:32:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Univariate Optimization]]"
Area: "[[Master's degree]]"
---
# Fastest Local Optimization

When dealing with algorithms like the Dichotomic Search, choosing the next point $x$ "right in the middle" of the interval is just the simplest approach. It is obviously better if $x$ is close to the actual minimum $X_*$. Ideally, if we could perfectly guess $x=x_*$, the algorithm would stop in one single iteration.

To improve our guess, we can use the fact that we already know a lot about the function $f$ at the boundaries of our interval: we know $f(x_-)$, $f(x_+)$, $f'(x_+)$, and $f'(x_-)$. The powerful general idea here is to construct a model of $f$ based on this known information.

## Improving the Dichotomic Search: Interpolation

We can build a quadratic interpolation, forming a parabola $ax^2 + bx + c$ that "agrees" with $f$ at the boundaries $X_+$ and $X_-$. We have three parameters ($a, b, c$) but four conditions (two function values, two derivative values). Since something's gotta give, we have three possible cases to build this model.

One way is to match the derivatives at the boundaries:
$$2ax_+ + b = f'(x_+)$$
$$2ax_- + b = f'(x_-)$$

Solving this system gives us the parameters for our parabola:
$$a = \frac{f'(x_+) - f'(x_-)}{2(x_+ - x_-)}$$
$$b = \frac{x_+f'(x_-) - x_-f'(x_+)}{x_+ - x_-}$$

To find the minimum of this new quadratic model, we solve $2ax + b = 0$ (the constant $c$ is irrelevant for the derivative). This yields our new estimated point:
$$x = \frac{x_-f'(x_+) - x_+f'(x_-)}{f'(x_+) - f'(x_-)}$$

This specific approach is known as the "method of false position", a.k.a. "secant formula". By construction, this new $x$ is always structurally in the middle between $x_+$ and $x_-$.
*(Exercise: develop the other cases of quadratic interpolation and discuss them)*

### The Map is not the World
A very general issue in optimization is that the model is merely an estimate. In this case, the quadratic model can be "very skewed".

For instance, depending on the slopes:
- $f'(x_+) \gg -f'(x_-) \Rightarrow x \approx x_-$
- $f'(x_+) \ll -f'(x_-) \Rightarrow x \approx x_+$

These are wrong, bad choices because they can lead to very short steps, which in turn causes slow convergence.
The general remedy is to never completely trust the model: we must regularise and stabilise the step. In this specific case, we impose a minimum guaranteed decrease $\sigma \le 0.5$ (a safeguard). We force the next point to be at least a fraction $\sigma$ inside the interval:
$$x \leftarrow \max\{x_- + \sigma(x_+ - x_-), \min\{x_+ - \sigma(x_+ - x_-), x\}\}$$

In the worst case, this safeguarded approach guarantees linear convergence with a rate $r = 1 - \sigma$. Hopefully, it is (much) faster than that when the model is "right".

## Theory & More Interpolation

Does building these models really show in practice? And how much faster can it get?
Quadratic interpolation has superlinear convergence if started "close enough" to the minimum.

Specifically, if $f \in C^3$, $f'(x_*) = 0$ and $f''(x_*) \ne 0$, then there exists a $\delta > 0$ such that:
$$x^0 \in [x_* - \delta, x_* + \delta] \Rightarrow \{x^i\} \to x_* \text{ with } p = (1+\sqrt{5})/2$$
*(Don't you just love maths? $1 < p = \phi \approx 1.618 < 2$, which is the golden ratio).*

This proves it is "very fast" already, but can we make it even faster?
If we use all four conditions we have, we can fit a cubic polynomial and use its minima. While rather tedious to write down, analyse, and implement, it theoretically pays off: cubic interpolation has quadratic convergence ($p = 2$) and seems to work pretty well in practice.
*(Exercise: not for the faint of heart, develop cubic interpolation)*

## Newton's Method

A better model of $f \equiv f'$ leads to a better guess of $X_*$, which translates to a faster algorithm. Building a better model requires either using more points or using more (higher-order) derivatives.

Newton's method (or tangent method) uses a first-order model of $f'$ (the tangent line of the derivative) at the current point $x^i$:
$$L'_i(x) = L'_{x^i}(x) = f'(x^i) + f''(x^i)(x - x^i) \approx f'(x)$$

To find the minimum, we solve for the root of this linear approximation, setting $L'_i(x) = 0 \approx f'(x) = 0$:
$$x = x^i - f'(x^i)/f''(x^i)$$

*The iterative procedure for Newton's Method simply loops as long as the absolute value of the first derivative $|f'(x)|$ is strictly greater than the tolerance $\epsilon$. Inside the loop, the current guess is updated by subtracting the ratio between the first derivative and the second derivative: $x \leftarrow x - f'(x)/f''(x)$. A critical implementation detail to handle is what happens if $f''(x) = 0$, as it would cause a division by zero.*

An alternative view is that Newton's method minimizes the second-order Taylor model of $f$ at $x^i$:
$$Q_i(x) = Q_{x^i}(x) = f(x^i) + f'(x^i)(x - x^i) + f''(x^i)(x - x^i)^2/2$$
*(Note that Newton's is actually a general method to solve nonlinear equations).*

The main drawback is that it converges fast (at all!) only if started "close enough" to $X_*$. Otherwise, it would require globalization techniques.

## Mathematically Speaking: Newton's Method Proof

To rigorously prove the convergence rate, we rely on the Second-order Taylor's formula. For all $z$, there exists a point $w \in [x, z]$ such that:
$$f(z) - L_x(z) = f''(w)(z - x)^2/2$$
Simply put, the error of the linear approximation $L_x$ in $z$ is $(z-x)^2$ times the value of $f''$ somewhere in the middle.

**Hypotheses:** $f \in C^3$, $f'(x_*) = 0$ and $f''(x_*) \ne 0$.
**Thesis:** $\exists \delta > 0$ such that $x^0 \in [x_* - \delta, x_* + \delta] \Rightarrow \{x^k\} \to x_*$ with $p = 2$.

**Proof:**
We evaluate the distance to the optimum at the next iteration:
$$x^{i+1} - x_* = x^i - x_* + (f'(x_*) - f'(x^i))/f''(x^i)$$
$$= [f'(x_*) - f'(x^i) - f''(x^i)(x_* - x^i)]/f''(x^i)$$

Using Taylor's formula for $f'$, we know that $\exists w \in [x^i, x^*]$ such that:
$$f'(x_*) - f'(x^i) + f''(x^i)(x_* - x^i) = f'''(w)(x_* - x^i)^2/2$$

Substituting this back into our error equation gives:
$$x^{i+1} - x_* = [f'''(w)/2f''(x^i)](x^i - x_*)^2$$

Because the functions are continuous, there exists a $\delta > 0$ such that $|f''(x)| \ge k_2 > 0$ and $|f'''(w)| \le k_1 < \infty$. Therefore:
$$\forall x, w \in [x_* - \delta, x_* + \delta] \Rightarrow |x^{i+1} - x_*| \le [k_1/2k_2](x^i - x_*)^2$$

As long as we start close enough so that the scaling factor ensures a contraction ($k_1(x^i - x_*)/2k_2 \le 1$), the error strictly decreases:
$$|x^{i+1} - x_*| < |x^i - x_*|$$

This guarantees that the sequence $\{x^i\} \to x_*$, and since the new error is proportional to the square of the old error, the convergence is indeed quadratic.
# References