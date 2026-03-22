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
# Newton's Method

A **better model** of $f \equiv f'$ leads to a better guess of $X_*$, which translates to a faster algorithm. Building a better model requires either using more points or using more (higher-order) derivatives.

Newton's method (or tangent method) uses a **first-order model** of $f'$ (the tangent line of the derivative) at the current point $x^i$:
$$L'_i(x) = L'_{x^i}(x) = f'(x^i) + f''(x^i)(x - x^i) \approx f'(x)$$

To find the minimum, we solve for the root of this linear approximation, setting $L'_i(x) = 0 \approx f'(x) = 0$
$$x = x^i - f'(x^i)/f''(x^i)$$
![[Pasted image 20260303202558.png]]

```
procedure x = NM(f, x, epsilon)
	while(|f'(x)| > epsilon) do
		x <- x - f'(x)/f''(x) // what if f''(x) = 0
```

The iterative procedure for Newton's Method simply loops as long as the absolute value of the first derivative $|f'(x)|$ is strictly greater than the tolerance $\epsilon$. Inside the loop, **the current guess is updated by subtracting the ratio between the first derivative and the second derivative**: $x \leftarrow x - f'(x)/f''(x)$. A critical implementation detail to handle is what happens if $f''(x) = 0$, as it would cause a division by zero.

An alternative view is that Newton's method minimizes the **second-order Taylor model** of $f$ at $x^i$:
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