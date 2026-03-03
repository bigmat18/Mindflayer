---
Data: 2026-02-27T13:07:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Univariate Optimization]]"
Area: "[[Master's degree]]"
---
# Local Optimization

When dealing with optimization problems, finding the absolute lowest point (the global minimum, $f_*$) is generally impossible because isolated minima can be anywhere. Even if we stumbled upon $X_*$, recognizing it as the global optimum is the "really difficult thing".

For this reason, we resort to a **weaker**, much more practical condition: **Local Optimization**. 
$X_*$ is a **local minimum** if:
$$x_* = \text{argmin}\{f(x) : x \in X(x_*, \epsilon) = [x_* - \epsilon, x_* + \epsilon]\}$$
for some $\epsilon > 0$.

A stronger notion is the **strict local minimum**, where the point is strictly lower than all its neighbors: $f(x_*) < f(z) \forall z \in X(x_*, \epsilon) \setminus \{x_*\}$.

Why is this useful? Because near $X_*$, the function $f$ typically has a predictable shape. If $f$ is unimodal on an interval $X = [x_{-}, x_{+}]$, it is strictly decreasing before the minimum and increasing after it.

![[Pasted image 20260302184101.png | 500]]

While most functions are not globally unimodal, they *are* unimodal if you restrict your focus to the **attraction basin** of $X_*$.

![[Pasted image 20260302184124.png | 500]]

Unfortunately, all local optima "look the same" (including the global one). 

![[Pasted image 20260302184145.png | 500]]

This makes finding *some* local optimum much easier, but finding the *right* (global) one remains entirely different.

![[Pasted image 20260302184236.png | 500]]

## The Guide: First-Order Model
To efficiently find a local minimum, we need a guide that tells us which direction $f$ is decreasing. For a simple linear function $f(x) = bx [+c]$, you just go left if $b > 0$ and right if $b < 0$.

![[Pasted image 20260302184534.png | 320]]    ![[Pasted image 20260302185010.png | 320]]

For nonlinear functions, we use the **first-order model** of $f$ at a point $x$:
$$L_x(z) = f'(x)(z - x) + f(x)$$
This is the best linear approximation (the tangent line) of $f$ at $x$, meaning $L_x(z) \approx f(z) \forall z \in [x-\epsilon, x+\epsilon]$ for some small $\epsilon > 0$.

Here, the trusty first derivative $f'(x)$ acts as our slope:
- If $f'(x) < 0 \Rightarrow f$ is decreasing at $x$.
![[Pasted image 20260302185112.png | 300]]

- If $f'(x) > 0 \Rightarrow f$ is increasing at $x$.
![[Pasted image 20260302185131.png | 300]]

- If $X_*$ is a local minimum, then $f'(x_*) = 0$. This is a **stationary point** (or a root of $f'$).
![[Pasted image 20260302185152.png | 300]]

## Mathematically Speaking: Derivatives and Smoothness
The formal definition of the derivative is:
$$f'(x) = \lim_{t \to 0} \frac{f(x+t) - f(x)}{t}$$
This limit must be finite and exist. For the limit to exist, the left and right derivatives must be equal and finite:
$$f'_{-}(x) = \lim_{t \to 0_{-}} \frac{f(x+t) - f(x)}{t}$$
![[Pasted image 20260302190012.png | 300]]

$$f'_{+}(x) = \lim_{t \to 0_{+}} \frac{f(x+t) - f(x)}{t}$$
![[Pasted image 20260302190029.png | 300]]

Non-differentiable functions happen in practice. For example, $f(x) = |x| = \max\{x, -x\}$ has $f'(x) = -1$ if $x < 0$ and $f'(x) = +1$ if $x > 0$, but at $x=0$, $f'(x)$ is undefined. 
Remember: if $f$ is differentiable at $x$, it is continuous at $x$, but the reverse does not hold.

- limite must be **finite**
![[Pasted image 20260302185333.png | 300]]

- and it **exists at all**
![[Pasted image 20260302185412.png | 300]]

**Differentiability Classes:**
- $f' \in C^0 \equiv f \in C^1 \equiv f$ continuously differentiable $\Rightarrow f \in C^0$.
- $f'' \in C^0 \equiv f \in C^2 \equiv f' \in C^1 \Rightarrow f' \in C^0 \Rightarrow f \in C^1 \Rightarrow f \in C^0$.

If a function $f \in C^1$ is globally  **[[Optimization Difficult#Lipschitz Continuity|Lipschitz continuous]] (L-c)** on $X$, its slope is bounded: $|f'(x)| \le L \forall x \in X$. The best possible scenario is having a function in $C^2$ on a finite set $X$, meaning both the function and its derivative are globally L-c, preventing the algorithm from encountering infinitely steep spikes.

## Classifying Stationary Points
If $f'(x) < 0$ or $f'(x) > 0$, $x$ clearly cannot be a local minimum. 

![[Pasted image 20260302191237.png | 400]]

Hence, $f'(x) = 0$ is a requirement for all local/global minima. 

![[Pasted image 20260302191249.png | 400]]

However, $f'(x) = 0$ is also true for local maxima and saddle points. To tell them apart, we must look at the second derivative $f'' = [f']'$ (the curvature).

![[Pasted image 20260302191305.png | 400]]

**A Polynomial Example:**
Consider the following complex function:
$$f(x) = \frac{91}{30}x^2 - \frac{19}{6}x^3 - \frac{54}{25}x^4 + \frac{93}{23}x^5 - \frac{23}{36}x^6 - \frac{121}{93}x^7 + \frac{72}{91}x^8 - \frac{13}{74}x^9 + \frac{9}{640}x^{10}$$
![[Pasted image 20260302190130.png]]

The roots of its first derivative $f'$ are the "interesting" points:
$$f'(x) = \frac{91}{15}x - \frac{19}{2}x^2 - \frac{216}{25}x^3 + \frac{465}{23}x^4 - \frac{23}{6}x^5 - \frac{847}{93}x^6 + \frac{576}{91}x^7 - \frac{117}{74}x^8 + \frac{9}{64}x^9$$
![[Pasted image 20260302190144.png]]

And the sign of $f''$ (if not zero) tells the maxima apart from the minima:
$$[f'(x)]' = \frac{91}{15} - 19x - \frac{648}{25}x^2 + \frac{1860}{23}x^3 - \frac{115}{6}x^4 - \frac{1694}{31}x^5 + \frac{576}{13}x^6 - \frac{468}{37}x^7 - \frac{81}{64}x^8$$
![[Pasted image 20260302190202.png]]

# References