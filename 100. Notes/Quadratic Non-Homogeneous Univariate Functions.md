---
Data: 2026-02-17T19:17:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Quadratic Non-Homogeneous Univariate Functions

The natural next step is $f(x) = a x^2 + b x$ (non-homogeneous quadratic), with parameters $(a, b) \in \mathbb{R}^2$. This is essentially a homogeneous quadratic plus a linear term.

- **Key Observation:** $\min \{ a x^2 + b x \} \ne \min \{ a x^2 \} + \min \{ b x \}$ (no additivity of minima in general).
- **Roots:** Always $x=0$ and $x = -b/a$ (unless $a=0$).
- **General Strategy:** If $f(x)$ is complicated, simplify it by **changing the variable space** (reformulation). This is only needed if both $a \ne 0$ and $b \ne 0$ (otherwise reduces to previous cases).

### Optimization
Define the **vertex** (shift point) $\bar{x} = -b/(2a)$ and set $z = x - \bar{x} \equiv x = z + \bar{x}$.
- **Algebraic Derivation:**
$$f(x) = a(z + \bar{x})^2 + b(z + \bar{x}) = a z^2 + 2 a z \bar{x} + a \bar{x}^2 + b z + b \bar{x}$$
$$= a z^2 + (2 a \bar{x} + b) z + (a \bar{x}^2 + b \bar{x})$$
    By choice of $\bar{x}$, the linear term vanishes: $2 a \bar{x} + b = -b + b = 0$. Thus:
$$f(x) = a z^2 + f(\bar{x}) = g(z)$$
	The function is now a **homogeneous quadratic** translated horizontally by $\bar{x}$ and vertically by $f(\bar{x})$.
    
- **Optimization:** The argmin/argmax of $g(z)$ (depending on sign of $a$) is $z=0$, so $x = \bar{x}$ in original coordinates. Then, apply the results from Section 3 to $g(z)$.
    
- **Closed-Form Solution:** $O(1)$ again—solve for $\bar{x}$ and evaluate.
    
This approach (shifting to center the minimum) generalizes to multivariate quadratics.

# References