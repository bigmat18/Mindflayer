---
Data: 2026-02-18T19:55:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Introduction to Optimization]]"
Area: "[[Master's degree]]"
---
# Gradiant Method

When dealing with the optimization of a  [[Homogeneous (general case)|quadratic function]] , if we are lucky, the problem is equivalent to solving the linear system $Q\bar{x} = -q$. Solving this system via direct methods (such as Gaussian elimination) has a computational cost of $O(n^3)$ in the worst case. 

This approach is feasible for moderate dimensions, around $n \approx 10000$ (ignoring memory limits), but becomes impractical for very large dimensions (e.g., $n \approx 10^9$).

For this reason, we resort to **iterative procedures**: starting from an initial guess $x_0$, we generate a sequence $x_i \rightsquigarrow x_{i+1}$ that should "go towards an optimal solution".
The natural way to view this process is to observe the sequence of function values $\{f_i = f(x_i)\}$, which must tend towards the optimal value $f_*$.

Typically, **we cannot obtain $f_*$ in finite time** (i.e., there is no $i$ such that $v_i = f_*$), but we can get as close as we want: convergence happens "in the limit".

Recall that a sequence $\{v_i\} \to v$ if $\lim_{i\to\infty} v_i = v$ that means
$$
\forall \epsilon > 0 \exists h \text{ s.t. } |v_i -v| \leq \epsilon \forall i \geq h
$$
$$
\lim_{i\to \infty} v_i = +\infty \Leftrightarrow \forall M > 0 \exists h \text{ s.t. } v_i \geq M \forall i \geq h
$$
A sequence $\{x_i\}$ such that $\{f_i\} \to f_*$ is defined as a **minimizing sequence**. It is important to note that a sequence might not have a limit, but any **monotone sequence** admits a limit; for this reason, algorithms that guarantee monotonicity (i.e., $f_{i+1} < f_i$) are preferable.

## Basic Idea of the Gradient Method
We generally assume a minimization problem (maximization is equivalent).
Given a point $x_i$, we necessarily compute the gradient $g_i = Qx_i + q$.
* If $g_i = 0$, we have found the optimum and stop.
* In floating-point arithmetic, the exact condition "$g_i = 0$" is not feasible, so we stop when $\|g_i\| \le \epsilon$ (for some chosen $\epsilon$).

The fundamental idea is: if $\|g_i\| > 0$, we can produce a point $x_{i+1}$ that is "better" than $x_i$.
To do this, we consider the "[[Picturing multivariate functions (Tomography)]]" (a 1D section of the function) along the direction of the anti-gradient:
$$\phi_{x_i, -g_i}(\alpha) = f(x_i - \alpha g_i) - f(x_i)$$
- $x_i$ is my center that is the current value in gradiant descent
- $-g_i$ its the negative direction to descent. 

Expanding the quadratic function, we obtain:
$$= \frac{1}{2}(x_i - \alpha g_i)^T Q (x_i - \alpha g_i) + q(x_i - \alpha g_i) - f(x_i)$$
$$= \frac{1}{2}\alpha^2 (g_i)^T Q g_i - \alpha [(g_i)^T Q x_i + q g_i]$$
$$= \frac{1}{2}\alpha^2 (g_i)^T Q g_i - \alpha \|g_i\|^2$$
- $\frac{1}{2}\alpha^2 (g_i)^T Q g_i$ is positive
- $- \alpha \|g_i\|^2$ is negative

from si formula we can calcuiltate the sstepsize $\alpha$

For some $\alpha > 0$, we have $\phi_{x_i, -g_i}(\alpha) < 0$, which implies $f(x_i - \alpha g_i) < f(x_i)$.
The same information (called **gradient**) telling us "you cannot stop" is simultaneously telling us "you can get a better solution by moving in that direction". This immediately suggests a monotone algorithm.

## The Algorithm for Quadratic Functions
In the quadratic case, it is easy to minimize the one-dimensional function $\phi_{x_i, -g_i}(\alpha)$ to find the optimal step (*Exact Line Search*):
$$\alpha_i = \frac{\|g_i\|^2}{(g_i)^T Q g_i}$$
- $\|g_i\|^2$ is the normalization of gradiant to do a stepsize propozionato to g.
- $(g_i)^T Q g_i$ its local curvature if we have an high curvature the stepsize is small

One can verify that the value of $\alpha$ is bounded by the eigenvalue spectrum: $1/\lambda_1 \le \alpha \le 1/\lambda_n$.

Computing $g_i$ and the optimal value of $\alpha$ costs $O(n^2)$ (dominated by the matrix-vector product). If $n$ is large, this is much more efficient than $O(n^3)$, allowing us to perform many iterations.

```pseudo 
procedure x = GMQ(Q, q, x, epsilon) // data of the problem
	do forever 
		g <- Qx + q; 
		if ( ||g|| <= epsilon ) then break; 
		alpha <- stepsize();
		x <- x - alpha * g;
		
		
stepsize() { return ||g||² / (g^T Q g); }
```

This algorithm is very simple, but special cases must be considered: what happens if the denominator in the stepsize is zero? What if $Q \not\succeq 0$? Furthermore, is it possible to rewrite the code to perform only one product with $Q$ per iteration to maximize efficiency?

This happens when $(g_i)^T Q g_i$ is 0, this is the curvature of own tomography. 
## Convergence Analysis ($Q \succ 0$)
Using the optimal stepsize yields an important geometric property: successive gradients are orthogonal, i.e., $g_{i+1} \perp g_i$. This generates a "zig-zag" path.

Let's consider the "homogeneous form of the error":
$$A(x) = \frac{1}{2}(x - x_*)^T Q (x - x_*)$$
this function measure how much you are distant form the minimum $x_*$, "pesando" the distance with matrix $Q$. But we dont know $x_*$ and for this we use a different version.

Under the assumption that $Q \succ 0$ (positive definite), and using the relation $g_i = Q(x_i - x_*)$, the error update formula can be derived:
$$A(x_{i+1}) = \left( 1 - \frac{\|g_i\|^4}{((g_i)^T Q g_i)((g_i)^T Q^{-1} g_i)} \right) A(x_i)$$

It is easy to derive a convergence estimate using the **condition number** of $Q$, defined as $\kappa = \lambda_1 / \lambda_n$ (where $\lambda_1$ is the largest eigenvalue and $\lambda_n$ the smallest). Since:
$$\frac{\|x\|^4}{(x^T Q x)(x^T Q^{-1} x)} \ge \frac{\lambda_n}{\lambda_1} = \frac{1}{\kappa}$$
- $\lambda_1$ quanto è rapida la salita più forte
- $\lambda_n$ quanta è piatta la salita più dolce
- $k=1$ la valle è un cerchio perfetto, il gradient punta dritto al centro
- $k=1000$ la valle è un sigaro lunghissimo, gradient più tempo

We obtain:
$$A(x_{i+1}) \le \left( 1 - \frac{1}{\kappa} \right) A(x_i)$$
This means that when $k$ is very higth the errore will be reduce very little. This means the algorithm converges: $A(x_i) \le r^i A(x_0)$.

Since $r \le (\kappa - 1) / \kappa < 1$, the error $A(x_i) \to 0$ **exponentially fast** as $i \to \infty$.

An even better estimate is provided by the **Kantorovich inequality**:
$$r \le \left( \frac{\lambda_1 - \lambda_n}{\lambda_1 + \lambda_n} \right)^2 = \left( \frac{\kappa - 1}{\kappa + 1} \right)^2$$
this is the ratio between the highest and lowest eigenvector. 
## Complexity and Convergence Rates
Crucial sequencies:
- $\{x_i\} = \{ d_i = ||x_i - x_*||\}$ distanza fisica (euclidea) con il punto ottimale
- $\{f_i = f(x_i)\}$ valore della funzione
- $\{a_i = A(x_i)\}$ l'errore energetico (la distanza pesata della matrice $Q$)
- $\{r_i = R{x_i}\}$ il residuo (quanto è grande il gradiante)

The complexity of the algorithm is a function of the **prescribed accuracy** $\epsilon$. We seek the maximum number of iterations $k$ such that  $d_i / a_i / r_i \leq \epsilon \forall i \geq k$ the distance from the solution or the error is less than $\epsilon$.

The general formula is $v_k \le r^k v_1 \le \epsilon$, which leads to:
$$
k\geq [1 / \log(1/r)]\log(v_1 /\epsilon)
$$
- $1/r$ when $r\to 1$ $\log{1} = 0$ and $1/0$ is $+\infty$ 
- more condition of $Q$ is large more iteration with get to achieve accuracy
$$ r \approx 1 \to k \in O\left( \frac{r}{1 - r} \log(v_1 / \epsilon) \right)$$
The good news is that this result is **dimension independent** ($n$ does not appear in the formula), making the method suitable for very large-scale problems.
The bad news is that the multiplicative constant tends to infinity as $r \to 1$.

$||x^{i} - x_{*}|| \leq \epsilon$ and $f(x^{i} - f_*) \leq \epsilon$ not the same ($\epsilon$):
- $a^{i} = \frac{1}{2}(x^{i} - x_{*})^T Q (x^{i} - x_*) \leq \epsilon \Rightarrow \lambda_n ||x^{i} - x_*||² \leq \epsilon$
- $\Rightarrow d^{i} = ||x^{i} - x_*|| \leq \sqrt{\epsilon / \lambda_n}$  this means distance is different than value, if We stop algorithm when $\epsilon$ is small potremmo ancora fisicamente essere molto lontano dal centro, la divisione $\lambda_n$ gonfia l'errore.

In terms of speed classification (Rate of convergence), we define the limit:
$$\lim_{i\to\infty} \frac{f_{i+1} - f_*}{ (f_i - f_*)^p } = \frac{a_{i+1}}{(a_i)^p} \approx \frac{r^{i+1}}{(r_i)^p} = r \:\:\:\: \begin{cases} x^p \to 0 & \text{faster than}\\ x\to 0 & \text{when } p > 1 \end{cases}$$

* **Linear Convergence ($p=1, r < 1$):** Error decreases as $r^i$, so $i \in O(\log(1/\epsilon))$. Considered "good" unless $r \approx 1$.
* **Sublinear Convergence ($p=1, r = 1$):** Error decreases very slowly (e.g., $O(1/i)$ or $O(1/\sqrt{i})$). Implies $i \in O(1/\epsilon)$ or worse.
* **Quadratic Convergence ($p=2, r > 0$):** Error decreases as $1/2^{2^i}$, so $i \in O(\log(\log(1/\epsilon)))$. In practice, the number of correct digits doubles at each iteration.
* **Superlinear Convergence ($p\in (1,2), r = 0$)**: this is something in the middle.

The gradient method on strictly convex quadratic functions exhibits linear convergence.

![[Pasted image 20260220111330.png | 450]]

![[Pasted image 20260220111348.png | 450]]

![[Pasted image 20260220111449.png | 450]]

Green and red are super-linear convergence in particolar red line is convergence with Newton method.
![[Pasted image 20260220111430.png | 450]]

## Stopping Criteria and Ill-Conditioning
The ideal stopping criterion would be $A(x_i) \le \epsilon$ (absolute error) or $R(x_i) \le \epsilon$ (relative error), but $f_*$ is typically unknown. 
We use $\|g_i\|$ as a "proxy" for $A(x_i)$, hoping that if the gradient is small, the error is also small. However, the exact relationship is not trivial. If the function is very "flat" (small eigenvalue $\lambda_n$), we might have a small gradient even if we are still far from the solution.
$$
g_i = Q(x_i - x_*) \Rightarrow ||g_i|| \leq \lambda_1 ||x_i - x_*||... \text{ wrong inequality}
$$
$$
||g_i|| \leq \epsilon \nRightarrow ||x_i - x_*|| \text{ "small"}
$$
- if we know $\delta \geq ||x^{i} - x_*||$ which we don't, then $||g^{i}|| \leq 2\epsilon / \delta \Rightarrow a^{i} \leq \epsilon$  
- if we know $\lambda_n > 0$ which we don't, $||g^{i}|| \leq \sqrt{2 \lambda_n \epsilon} \Rightarrow a^{i} \leq \epsilon$

A critical issue is when "exponentially fast" is not "really fast". Convergence is fast if $\lambda_1 \approx \lambda_n$ (circular level sets), but very slow if $\lambda_1 \gg \lambda_n$, i.e., when $\kappa = \lambda_1 / \lambda_n \to \infty$ ($Q$ is **ill-conditioned**) $\Rightarrow r \to 1 \Rightarrow$ slow.
Geometrically, if the level sets are very elongated ellipses and $g_{i+1} \perp g_i$, the algorithm performs many "zig-zags".

For example, if $\kappa = 1000$, we get $r \approx 0.996$. To reduce the error by a factor of $10^{-6}$, $k \ge 3450$ iterations might be needed.

More bad news:
- $\lambda_1$ and $\lambda_2$ **may depend on n**, $k$ may grow as $n\to \infty$
- behavior in practice is very close to this theoretical bound.
- $\lambda_n = 0 \equiv k=\infty$ happens

## Semidefinite Case ($\lambda_n = 0$)
What happens if $\lambda_n = 0$? (The matrix $Q$ is positive semidefinite; the function is convex but not strictly convex).

It does not mean it doesn't converge, but we cannot prove the exponential convergence seen above.
In this case, it is proven that:
$$f(x_i) - f_* \le \frac{2\lambda_1 \|x_1 - x_*\|^2}{i - 1}$$
The complexity becomes $k \ge \frac{2\lambda_1 d}{\epsilon}$.
This is a case of **sublinear convergence** ($O(1/\epsilon)$ vs. $O(\log(1/\epsilon))$), which is exponentially slower. Obtaining high precision becomes unfeasible.

To improve performance, especially in the ill-conditioned case, the fundamental idea will be to **change the space** (Preconditioning), a concept that will be explored later.
# Reference