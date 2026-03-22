---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# Gradient Method with Inexact Line Search

When tackling the optimization of general functions, finding the exact optimal step size along a descent direction (Exact Line Search) is, in most cases, impossible or computationally too expensive. For this reason, we resort to **Inexact Line Search** procedures, where we do not look for the perfect minimum, but settle for a step $\alpha$ that decreases the function "enough".

## The Armijo Condition

The first problem to solve is preventing the algorithm from taking steps that are too long, which could cause it to diverge. To guarantee a sufficient decrease, we introduce the Armijo condition, based on a parameter $0 < m_1 < (\ll) 1$:

$$(A) \quad \varphi(\alpha) \le \varphi(0) + m_1 \alpha \varphi'(0)$$

- $\varphi(\alpha)$ represents the value of our function moving by a step $\alpha$ along the descent direction.
    
- $\varphi(0) + m_1 \alpha \varphi'(0)$ is the equation of a line (a first-order model) passing through the starting point. The original slope $\varphi'(0)$ is "relaxed" by multiplying it by a very small fraction $m_1$ (often $m_1 \approx 0.0001$).
    
- In practice, this condition requires that the function decreases by at least a small fraction ($m_1$) of the descent that was "promised" by the initial derivative at the starting point.
    

If we enforce that the step is never smaller than a certain threshold ($\alpha^i \ge \overline{\alpha} > 0$) and condition (A) holds at every iteration $i$, the algorithm works: every accumulation point of the sequence $\{x^i\}$ is a stationary point (where $\{||\nabla f(x^i)||\} \to 0$), unless the function diverges to $-\infty$. In fact, the decrease at each step is guaranteed by the inequality:

$$f^{i+1} \le f^i + m_1 \alpha^i \varphi_i'(0) \le f^i - m_1 \overline{\alpha} \epsilon$$

Applying this logic iteratively yields $f^i \le f^0 - m_1 \overline{\alpha} \epsilon i \Rightarrow \{f^i\} \to -\infty$.

## The Wolfe Condition

While Armijo prevents steps that are too long, we need a rule to avoid steps that are _too short_, otherwise the algorithm might stall far from the solution. To solve this problem, we introduce a second parameter $m_2$ such that $m_1 < m_2 < 1$ and formulate the Wolfe Condition:

$$(W) \quad \varphi'(\alpha) \ge m_2 \varphi'(0)$$

- This condition requires the derivative at the new point $\alpha$ to be "closer to 0" (less steep) than the starting derivative. If the slope is still very negative, it means there is still plenty of room to descend and the step taken was too timid.
    

There is also a stricter version, the **Strong Wolfe condition**:

$$(W') \quad |\varphi'(\alpha)| \le m_2 |\varphi'(0)| [cite_start]= -m_2 \varphi'(0) \Rightarrow (W)$$

- By taking the absolute value, we force the derivative to be numerically small. The intersection $(A) \cap (W')$ ensures that $\varphi'(\alpha) \gg 0$, helping the algorithm to discard some local maxima and capture the desired local minima.
    

## Mathematical Proof (Rolle's Theorem)

One might wonder: does there always exist a step $\alpha$ that simultaneously satisfies both Armijo and Wolfe? The answer is yes, and it is proven using Rolle's Theorem.

Assuming $\varphi \in C^1$ and that $\varphi(\alpha)$ is bounded below for $\alpha \ge 0$ , we define the distance between the Armijo limit line $l(\alpha)$ and our function:

$$d(\alpha) = l(\alpha) - \varphi(\alpha)$$

$$d'(\alpha) = m_1 \varphi'(0) - \varphi'(\alpha)$$

We know that $d(0) = 0$. Since the function is bounded below, there will be a smallest point $\overline{\alpha} > 0$ where the curve "touches" the line again, meaning $d(\overline{\alpha}) = 0$. By Rolle's Theorem, if $d(0) = d(\overline{\alpha}) = 0$, there must exist an intermediate point $\alpha' \in (0, \overline{\alpha})$ where the derivative vanishes:

$$d'(\alpha') = 0 \equiv m_1 \varphi'(0) = \varphi'(\alpha')$$

Since we set $m_2 > m_1$, and knowing that $\varphi'(0)$ is negative, we deduce that:

$$m_2 \varphi'(0) < m_1 \varphi'(0) = \varphi'(\alpha')$$

This mathematically proves that the Wolfe condition $(W)$ is satisfied at the point $\alpha'$.

## Practical Implementation: Backtracking Line Search

Finding the exact point $\alpha'$ that satisfies both conditions can be complicated. A much simpler and widely used version in practice is the **Backtracking Line Search**, which exclusively checks the Armijo condition (A).

Plaintext

```
procedure \alpha = BLS(\varphi, \alpha, m_1, \tau)
    // \tau < 1
    while(\varphi(\alpha) > \varphi(0) + m_1 \alpha \varphi'(0)) do 
        \alpha <- \tau * \alpha;

http://googleusercontent.com/immersive_entry_chip/0
```
# References