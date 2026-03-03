---
Data: 2026-03-03T21:25:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Univariate Optimization]]"
Area: "[[Master's degree]]"
---
# Dichotomic Search
For linear or simple quadratic functions, setting $f'(x) = 0$ yields a closed formula ($x = -b/2a$). But for most transcendental or mixed functions, we need an algorithm to solve the nonlinear equation.

The **Dichotomic Search** (Bisection Method) relies on the intermediate value theorem: if $f'$ is continuous, and we have an interval where $f'(x_{-}) < 0$ and $f'(x_{+}) > 0$, there must be an $x \in [x_{-}, x_{+}]$ where $f'(x) = 0$.

```pseudo
procedure x = DS(f, x-, x+, epsilon)
	do forever                // invariant: f'(x-) < -epsilon, f'(x+) > epsilon
	x <- in_middle_of(x-, x+); 
	compute f'(x);
	if (|f'(x)| <= epsilon) then break;
	if (f'(x) < 0) then x- <- x;
	else x+ <- x;
```

A trivial choice for the middle is the simple average of the boundaries. 

```
in_middle_of(x-, x+) { return (x+ + x-)/2; }
```

This algorithm features **linear convergence** with a rate $r = 0.5 < 0.618$, guaranteeing that the interval halves at every step. 
$$
k \approx 3.32 \log (D/\delta) < 4.78 \log(D / \delta)
$$

If $f'$ is L-smooth (having Lipschitz constant $L$), the number of iterations required to reach precision $\epsilon$ is roughly:
$$k \approx 3.32 \log(LD / 2\epsilon)$$

### Finding the Initial Interval
What if we don't know $x_{-}$ and $x_{+}$ such that the derivative changes sign?
We can dynamically expand our search space exponentially until we find the slopes we need.

```
delta_x <- 1;
while (f' (x+) <= -epsilon) do
	x+ <= x+ + delta_x;
	delta_x <- 2delta_x;
```

We do the same "in reverse" for $x_{-}$ (starting with a negative step size). This works in practice for all reasonable, coercive functions (where $\lim_{|x|\to\infty}f(x) = \infty$).

## Quadratic Interpolation
When dealing with algorithms like the **Dichotomic Search,** choosing the next point $x$ "right in the middle" of the interval is just the simplest approach. It is obviously better if $x$ is close to the actual minimum $X_*$. Ideally, if we could perfectly guess $x=x_*$, the algorithm would stop in one single iteration.

To improve our guess, we can use the fact that we **already know a lot about the function $f$ at the boundaries of our interval**: we know $f(x_-)$, $f(x_+)$, $f'(x_+)$, and $f'(x_-)$. The powerful general idea here is to construct a model of $f$ based on this known information.

We can build a **quadratic interpolation**, forming a parabola $ax^2 + bx + c$ that "agrees" with $f$ at the boundaries $X_+$ and $X_-$. We have three parameters ($a, b, c$) but four conditions (two function values, two derivative values). Since something's gotta give, we have three possible cases to build this model.

One way is to match the derivatives at the boundaries:
$$2ax_+ + b = f'(x_+)$$
$$2ax_- + b = f'(x_-)$$

Solving this system gives us the parameters for our parabola:
$$a = \frac{f'(x_+) - f'(x_-)}{2(x_+ - x_-)}$$
$$b = \frac{x_+f'(x_-) - x_-f'(x_+)}{x_+ - x_-}$$

To find the minimum of this new quadratic model, we solve $2ax + b = 0$ (the constant $c$ is irrelevant for the derivative). This yields our new estimated point:
$$x = \frac{x_-f'(x_+) - x_+f'(x_-)}{f'(x_+) - f'(x_-)}$$

This specific approach is known as the "method of false position", a.k.a. "secant formula". By construction, this new $x$ is always structurally in the middle between $x_+$ and $x_-$.

### The Map is not the World
A very general issue in optimization is that the model is merely an estimate. In this case, the quadratic model can be "very skewed".

For instance, depending on the slopes:
- $f'(x_+) \gg -f'(x_-) \Rightarrow x \approx x_-$
- $f'(x_+) \ll -f'(x_-) \Rightarrow x \approx x_+$

These are wrong, bad choices because they can **lead to very short steps**, which in turn causes slow convergence.

The general remedy is to never completely trust the model: we must regularise and stabilise the step. In this specific case, we impose a **minimum guaranteed decrease** $\sigma \le 0.5$ (a safeguard). We force the next point to be at least a fraction $\sigma$ inside the interval:
$$x \leftarrow \max\{x_- + \sigma(x_+ - x_-), \min\{x_+ - \sigma(x_+ - x_-), x\}\}$$

In the worst case, this safeguarded approach guarantees linear convergence with a rate $r = 1 - \sigma$. Hopefully, it is (much) faster than that when the model is "right".

Does building these models really show in practice? And how much faster can it get?
Quadratic interpolation has **superlinear** convergence if started "close enough" to the minimum.

Specifically, if $f \in C^3$, $f'(x_*) = 0$ and $f''(x_*) \ne 0$, then there exists a $\delta > 0$ such that:
$$x^0 \in [x_* - \delta, x_* + \delta] \Rightarrow \{x^i\} \to x_* \text{ with } p = (1+\sqrt{5})/2$$
*(Don't you just love maths? $1 < p = \phi \approx 1.618 < 2$, which is the golden ratio).*

This proves it is "very fast" already, **but can we make it even faster?** If we use all four conditions we have, **we can fit a cubic polynomial and use its minima**. While rather tedious to write down, analyse, and implement, it theoretically pays off: cubic interpolation has quadratic convergence ($p = 2$) and seems to work pretty well in practice.


# References