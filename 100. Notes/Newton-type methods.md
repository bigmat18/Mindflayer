---
Data: 2026-03-29T21:15:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Smooth Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
## Newton-Type Methods
If we want to find a better descent direction that leads to faster convergence, we must use a better model of the function. So far, we have relied on a linear model (the gradient). The logical next step is to move to a quadratic model.

### Newton's Method: The (Locally) Strictly Convex Case
When the function is strictly convex at the current point, its Hessian matrix (the matrix of second derivatives) is positive definite, meaning $\nabla^2 f(x^i) > 0$. This guarantees the existence of a unique minimum for our second-order Taylor approximation $Q_{x^i}(z)$.

To find this minimum, we use the Newton direction:
$$d^i = -[\nabla^2 f(x^i)]^{-1} \nabla f(x^i)$$
Instead of simply taking a step in the direction of steepest descent (the negative gradient), we multiply the gradient by the inverse of the Hessian matrix. The Hessian captures the curvature of the function. By doing this, we are effectively finding the exact minimum of the quadratic approximation $Q_{x^i}(z)$ in a single step.

Since this direction points exactly to the minimum of our model, there is no need to calculate the stepsize: we simply take the full step $\alpha^i = 1$.

The update rule for the pure Newton's method is simply:
$$x^{i+1} = x^i + d^i$$

Another way to interpret this method is to see it as solving a nonlinear equation. We want to find the point where the gradient is zero ($\nabla f(x) = 0$). We can approximate the gradient using a first-order Taylor expansion:
$$\nabla f(x) \approx \nabla f(x^i) + \nabla^2 f(x^i)(x - x^i)$$

Setting this expression equal to zero and solving for $x$, we obtain exactly the Newton step.

We also know this is guaranteed to be a descent direction. Since $\nabla^2 f(x^i) > 0$, its inverse is also positive definite ($[\nabla^2 f(x^i)]^{-1} > 0$). Therefore, the directional derivative is strictly negative:
$$\langle\nabla f(x^i), d^i\rangle = -\nabla f(x^i)^T [\nabla^2 f(x^i)]^{-1} \nabla f(x^i) < 0$$
*(Note: although it is negative, we still need to ensure it is "negative enough" to guarantee convergence).*


### (Global) Convergence of Newton's Method
Pure Newton's method is extremely fast, but it is not globally convergent: if you start too far from the minimum, it might diverge or oscillate. To remedy this, we create the Globalised Newton's method by simply adding an Armijo-Wolfe Line Search (AWLS) or a Backtracking Line Search (BLS), but always testing the ideal step $\alpha^0 = 1$ first.

There are three main theorems regarding its convergence:

###### Theorem 1 (Global Convergence):
If $f \in C^2$ is $L$-smooth and $\tau$-convex, the globalized method converges globally (via Zoutendijk's theorem). The descent angle is bounded:$$\cos(\theta^i) \le -\tau/L < 0$$**Explanation:** Because the eigenvalues of the Hessian are bounded between $\tau$ and $L$, the Newton direction can never become perfectly perpendicular to the gradient.

**Proof**: Theorem 1, two technical steps using $\nabla^2 f(x^i) d^i = -\nabla f(x^i)$:
$$\quad \langle \nabla f(x^i), d^i \rangle = -(d^i)^T \nabla^2 f(x^i) d^i \le -\tau \| d^i \|^2 \\[1ex] $$
$$ \| \nabla f(x^i) \| = \| \nabla^2 f(x^i) d^i \| \le \| \nabla^2 f(x^i) \| \| d^i \| \le L \| d^i \| \\[1.5ex] $$
$$\implies \cos(\theta^i) = \langle \nabla f(x^i), d^i \rangle / (\| \nabla f(x^i) \| \| d^i \|) \le -\tau / L $$
###### Theorem 2 (Quadratic Convergence)
If $f \in C^3$, at the optimum $\nabla f(x_*) = 0$, and the Hessian is positive definite $\nabla^2 f(x_*) > 0 \Rightarrow \exists \delta > 0$ such that if we start close enough to the optimum ($x^0 \in \mathcal{B}(x_*, \delta)$), the "pure Newton" ($\alpha^i = 1$) will converge to $x_*$ quadratically.

**Explanation:** Quadratic convergence means that the number of correct decimal places roughly doubles with every single iteration. It is a staggering speed.

**Proof**: basically same proof as for n = 1

###### Theorem 3 (The Transition)
If the sequence $\{x^i\} \to x_*$, then there will exist an iteration $h$ such that the full Newton step $\alpha^i = 1$ perfectly satisfies the Armijo condition (A) for all $i \ge h$. This requires the Armijo parameter to be $m_1 < 1/2$ (a larger $m_1$ would artificially reject the true minimum of a quadratic function).

**Explanation:** The algorithm naturally presents two phases. A "Global Phase" where the step $\alpha^i$ varies to navigate the space safely, automatically followed by a "Pure Newton Phase" where $\alpha^i = 1$ is always accepted, triggering quadratic convergence. This pure phase usually concludes the optimization in $O(1)$ ($\approx 6$) iterations in practice.

**Proof:** to understand why the full step $\alpha^i = 1$ is eventually accepted, we use the Taylor expansion of $f(x^i + d^i)$:
$$f(x^i + d^i) = f(x^i) + \langle\nabla f(x^i), d^i\rangle + \frac{1}{2}(d^i)^T [\nabla^2 f(x^i)] d^i + R(d^i)$$

Since $d^i$ is the Newton direction, we know that $\nabla^2 f(x^i) d^i = -\nabla f(x^i)$. Substituting this into the quadratic term we get:
$$= f(x^i) - \nabla f(x^i)^T [\nabla^2 f(x^i)]^{-1} \nabla f(x^i) + \frac{1}{2}\nabla f(x^i)^T [\nabla^2 f(x^i)]^{-1} \nabla f(x^i) + R(d^i)$$

Which elegantly simplifies to:
$$= f(x^i) + \frac{1}{2}\langle\nabla f(x^i), d^i\rangle + R(d^i)$$

As we approach the minimum, $d^i \to 0$. The directional derivative $\varphi_{x^i, d^i}'(0) = \langle\nabla f(x^i), d^i\rangle \to 0$, but the Taylor remainder $R(d^i)$ goes to 0 even faster. Eventually, the remainder is negligible, and the step produces exactly a fraction of $1/2$ of the promised descent. This is why the Armijo condition is satisfied as long as $m_1 < 1/2$.


### Geometric Interpretation: Newton = Gradient + Space Dilation
There is an incredibly elegant geometric interpretation of Newton's method: it is simply the standard Gradient Method operating in a distorted (dilated) space.

Consider a quadratic function $f(x) = \frac{1}{2}x^T Q x + qx$, where the Newton step is $d = -x - Q^{-1}q$. Taking this full step yields $\nabla f(x + d) = 0$, which means Newton's method terminates in exactly one iteration.

Since $Q > 0$ (positive definite), we can decompose it into $Q = R^T R$ (where $R$ is a non-singular matrix).
If we apply a bijective change of variables to "distort" our space, defining $z = Rx \equiv x = R^{-1}z$, our function becomes:
$$h(z) = f(R^{-1}z) = \frac{1}{2}z^T I z + q R^{-1}z$$

In this new "z-space", the Hessian is simply the Identity matrix $I$. The elliptical contour lines of the function have been stretched into perfect circles!
In this perfectly spherical space, the standard gradient $g = -\nabla h(z) = -z - R^{-1}q$ points exactly to the center. Taking a standard gradient step $\nabla h(z + g) = 0$ solves the problem instantly.

If we translate this magic gradient $g$ from z-space back to our original x-space, we get exactly the Newton direction:
$$R^{-1}g = R^{-1}(-z - R^{-1}q) = -x - Q^{-1}q = d$$

### The Non-Convex Case and Hessian Modifications
What happens if we are in a non-convex region where the Hessian is not positive definite? If $\nabla^2 f(x^i)$ has negative eigenvalues (e.g., at a saddle point), the Newton direction might point uphill, towards a maximum!

To solve the problem, we define a modified direction $d^i \leftarrow -[H^i]^{-1} \nabla f(x^i)$, where we force the matrix $H^i$ to be positive definite: $\tau I \le H^i \le LI$.

If $\nabla^2 f \ne 0$, we can choose a "small" $\epsilon^i$ to shift the matrix:
$$H^i = \nabla^2 f(x^i) + \epsilon^i I > 0$$

**Explanation:** By adding a multiple of the Identity matrix, we are adding $\epsilon^i$ to every eigenvalue. If we choose $\epsilon^i$ to be slightly larger than the absolute value of the most negative eigenvalue ($\lambda^n < 0$), all eigenvalues become positive.

A simple formula for this shift is $\epsilon = \max\{0, \delta - \lambda^n\}$ for a small appropriately chosen parameter $\delta$ (like 1e-8 or 1e-12). This approach perfectly solves the optimization problem $\min\{||H - \nabla^2 f(x^i)||_2 : H \ge \delta I\}$.

As the algorithm approaches a strict local minimum where $\nabla^2 f(x_*) \ge \delta I$, the shift $\epsilon^i$ naturally becomes 0, meaning $H^i = \nabla^2 f(x^i)$. The algorithm smoothly transitions back to pure Newton's method, regaining quadratic convergence in the tail.


### The Computational Bottleneck
Whether using the exact Hessian or a modified $H^i$, one still has to solve a linear system or compute a matrix factorization (like the Cholesky $H^i = L^i (L^i)^T$). This requires $O(n^3)$ operations. For large-scale problems (e.g., $n = 10^4+$), $O(n^3)$ is simply too expensive. We need something much cheaper, $O(n^2)$ or less, which leads us to the Trust Region approach and Quasi-Newton Methods.


### A Different Approach: Trust Region
The methods seen so far (Line Search) follow a two-phase approach: first, they choose a direction $d^i \in \mathbb{R}^n$, and then they search for a suitable step $\alpha^i \in \mathbb{R}$ along that direction. Trust Region (TR) methods completely flip this logic: first, you choose the maximum step length (the "trust radius" $\alpha^i$ or $r$), and only then do you search for the optimal direction within that radius.

#### The Negative Curvature Problem
In pure Newton's method, if $\nabla^2 f(x^i)$ has negative eigenvalues, there are directions of negative curvature along which $f$ decreases. These are exactly the places we want to go to minimize $f$, so why exclude them by modifying the Hessian?

The quadratic model $Q^i(z)$ does not have a global minimum over all $\mathbb{R}^n$ if there are negative curvatures. However, if we constrain the search to a compact set (a "trust region" $\mathcal{T}^i$ around $x^i$ where we know our quadratic model approximates the real function well), the minimum always exists.
$$x^{i+1} \in \text{argmin} \{Q^i(z) : z \in \mathcal{T}^i\}$$

**Explanation:** We are solving a constrained optimization problem at each iteration. If we choose a Euclidean sphere $\mathcal{B}_2(x^i, r)$ as the region $\mathcal{T}^i$, the problem is efficiently solvable ("round balls are simpler than kinky balls").

#### The Mathematical Solution of Trust Region
Replacing the exact Hessian with an approximation $H^i \approx \nabla^2 f(x^i)$ (not necessarily positive definite), the optimal point $x^{i+1} = x^i + d^i$ for the constrained quadratic submodel exists and is characterized by the following conditions (with $\exists \lambda^i \ge 0$):

1.  $H^i + \lambda^i I \ge 0$
2.  $||d^i|| \le r$
3.  $[H^i + \lambda^i I]d^i = -\nabla f(x^i)$
4.  $\lambda^i(r - ||d^i||) = 0$

**Explanation:**
* Equation 1 tells us that adding the scalar $\lambda^i$ to the diagonal of the matrix "corrects" the Hessian, forcing it to become positive semi-definite, elegantly solving the problem of negative curvatures.
* Equation 3 is a modified version of the Newton step.
* Equation 4 is the complementarity condition: if the computed step falls strictly within the radius ($||d^i|| < r$), then $\lambda^i = 0$ and we are taking a pure standard Newton step (the constraint has no effect). As the sequence converges to the optimum ($\{x^i\} \to x_*$), the step becomes small ($||d^i|| \to 0$), $\lambda^i = 0$ eventually, and quadratic convergence is recovered in the tail.


### Quasi-Newton Methods
Computing and inverting (or factorizing) the exact Hessian matrix $\nabla^2 f(x^i)$ costs $O(n^3)$ operations, which is impractical for large-scale problems. Quasi-Newton methods solve this problem by iteratively building a matrix $H^i$ that approximates the Hessian using only information gathered from the gradient in previous steps ("learning $\nabla^2 f$ out of samples of $\nabla f$").

The space of $H^i$ matrices that offer fast ("superlinear") convergence is large. Superlinear convergence is achieved if $H^i$ behaves like the true Hessian along the direction of the step just taken ($d^i$): we do not care if it is accurate elsewhere.

#### The Secant Equation
Let's define the difference between positions and the difference between gradients of two consecutive steps:
$$s^i = x^{i+1} - x^i = \alpha^i d^i$$
$$y^i = \nabla f(x^{i+1}) - \nabla f(x^i)$$

We want our new quadratic model $m^{i+1}(x)$ to agree with the newly observed derivative. Imposing the condition $\nabla m^{i+1}(x^i) = \nabla f(x^i)$, we obtain the secant equation:
$$(S) \quad H^{i+1} s^i = y^i$$

**Explanation:** This equation forces the new matrix $H^{i+1}$ to map the physical step $s^i$ exactly into the gradient change $y^i$. Multiplying on the left by $(s^i)^T$, we get the Curvature Condition:
$$(C) \quad \langle s^i, y^i \rangle = (s^i)^T H^{i+1} s^i > 0$$
*(Often written as $\rho^i = 1 / \langle s^i, y^i \rangle > 0$).*

**Explanation:** For $H^{i+1}$ to be positive definite, this dot product must be strictly greater than zero. Fortunately, if we use a Line Search that respects the strong Wolfe condition (W), the curvature condition (C) can always be satisfied.

#### DFP (Davidon-Fletcher-Powell)
To find $H^{i+1}$, we look for the matrix that satisfies (S), is positive definite ($H \ge 0$), and is "as close as possible" (minimizing the Frobenius distance $||H - H^i||_F$) to the previous matrix $H^i$. The solution is the DFP formula:
$$(DFP) \quad H^{i+1} = (I - \rho^i y^i (s^i)^T) H^i (I - \rho^i s^i (y^i)^T) + \rho^i y^i (y^i)^T$$

**Explanation:** Since we actually need the inverse $B^{i+1} = [H^{i+1}]^{-1}$ to compute the Newton step, we can apply the Sherman-Morrison-Woodbury (SMW) formula to update the inverse matrix directly:
$$(DFP^{-1}) \quad B^{i+1} = B^i + \rho^i s^i (s^i)^T - B^i y^i (y^i)^T B^i / (y^i)^T B^i y^i$$

Thanks to this formula, the update only requires matrix-vector products, reducing the computational cost to $O(n^2)$ per iteration, without ever having to compute an actual inverse.

#### BFGS (Broyden-Fletcher-Goldfarb-Shanno)
The DFP formula is quite efficient, but we can do better. By writing the secant equation for $B^{i+1}$ ($s^i = B^{i+1}y^i$) and minimizing the distance of the inverse, given that everything is symmetric (just swap $B \leftrightarrow H$ and $s \leftrightarrow y$), we obtain the BFGS formula:
$$(BFGS) \quad H^{i+1} = H^i + \rho^i y^i (y^i)^T - H^i s^i (s^i)^T H^i / (s^i)^T H^i s^i$$
$$(BFGS) \quad B^{i+1} = (I - \rho^i s^i (y^i)^T) B^i (I - \rho^i y^i (s^i)^T) + \rho^i s^i (s^i)^T$$

**Explanation:** BFGS builds an excellent trade-off between the cost per iteration (which remains $O(n^2)$) and the speed of convergence.

#### Limited-Memory BFGS (L-BFGS)
For truly large problems (e.g., very high $n$), even $O(n^2)$ memory/time to save the dense matrix $B$ is far too much.

The solution is L-BFGS ("Limited-memory BFGS"): instead of explicitly storing the matrix $B^i$, we "unroll" the last iterations, keeping in memory only the last $k$ update vectors $s$ and $y$ (with $k \ll n$).

Defining $V^i = I - \rho^i y^i (s^i)^T$, the update takes the form:
$$B^{i+1} = (V^{i-k}V^{i-k+1}...V^i)^T B^{i-k} (V^{i-k}V^{i-k+1}...V^i) + \dots + \rho^i s^i (s^i)^T$$

**Explanation:** When we need to compute the step $d^i = -B^i \nabla f(x^i)$, we reconstruct the result iteratively by applying the saved vector-vector products. The cost plummets to $O(kn)$ per iteration. There is a trade-off: as $k$ increases, it converges like Newton's method; as $k$ decreases, the convergence worsens and behaves like the gradient method.

# References