---
Data: 2026-05-20T23:59:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Level Stabilization (PLBM)

The idea of level stabilitazion is in some sense opposite to that of the prevous approaches. In gereral $f_{\mathcal{B}}$  **is too optimisitc a model of** $f$, in that it **underestimates the true value of $f$** in a large part of the space.

So with the PLBM we do the opposite, it fix beforehand **how much descent the model should attain**, so the algorithms will work in the **sublevel set** $lev(f_{\mathcal{B}}, l)$ for some given level parameter $l<f(\bar{x})$ 

We need to change the Mater Problem to select the right point:
$$
x^* = \arg\min \{ ||x-x^*_i|| : f_i(x) \leq l_i \}
$$

The **advantage** of PLBM approach in that the stabilization parameter $l$ has the scale of function valurs, which may make it easier to choose.

### Choose the $l_i$ value
##### $f_*$ is know
This is the simple case. The actual target $l_i$ must be between $[f(x^*_i), f_i]$ that means between the actual record ($f(x_i^*)$) and the global minimum. The simple strategy is to use a parameter $\lambda \in (0,1]$ 
$$
l_i = \lambda f(x_i^*) + (1 - \lambda) f_*
$$
##### $f_*$ is unknown
This is own case, the most common one. We can use the same formulaton above replacing $f_*$ with its lower bound $v_i$ that we could obtain computer the base master problem.

This means that we need to solve **two times the Master Problem at each iteration**. This could be a good strategy only if the oracle computation is an eavier computation compered to master problem.

##### $l_i$ arbitrarily
The alternative to the above problem is to choose $l_i$ arbitrarily. The possible troubling consegune is that **we will be too much optimistic and we choose $l_i$ impossible to achive**, this means $l_i < f_*$

In optimiziation the Master Problem may be empty but this is not an issues because it does not bring to a crash, it is a discovery, we now know that the $l_i$ value is a valid lower limit for $f_*$. So the algortihm can update $l_{i+i} > l_{i}$ and iterate

## Convergence

This section demonstrates the theoretical soundness of the Proximal Level Bundle Method (PLBM). The goal is to mathematically prove that the algorithm cannot fail (it will always reach the target) and to estimate the maximum time required to do so.

### The Stopping Condition (Duality Gap)
The PLBM possesses a mathematical stopping guarantee, based on the closure of the **Duality Gap**:
$$\overline{f}^k - v^k \le \epsilon$$
- **$\overline{f}^k$ (Upper Bound):** The current record, which is the lowest real error found so far by evaluating the neural network (the "ceiling").
- **$v^k$ (Lower Bound):** The absolute theoretical minimum calculated by the approximated piecewise-linear model (the "floor").

When the difference between the ceiling and the floor drops below a microscopic tolerance threshold $\epsilon$, the algorithm stops. This guarantees the absolute certainty of having found the exact global minimum.

### The Convergence Hypotheses
The global convergence theorem requires the objective function to satisfy two mandatory geometric requirements.
##### A. Convexity
The cost function must have the shape of a "bowl" with a single global minimum point, without fake holes (local minima). The Mean Squared Error (MSE) is inherently convex, and the addition of the $\ell_1$ norm strictly preserves this convexity.
##### B. Coercivity and Bounded Sublevel Sets
A function $f(w)$ is defined as **coercive** if:

$$\lim_{\Vert{}w\Vert{} \to \infty} f(w) = +\infty$$
- **Visual meaning:** Starting from the center and walking in any direction towards infinity, the altitude of the function will inexorably rise to positive infinity. There are no "flat corridors" to escape through without increasing the error.
- **The role of the $\ell_1$ norm:** The MSE alone might not be coercive (there is a risk of weights exploding to infinity without increasing the error). The regularization term $\lambda \Vert{}w\Vert{}_1$ acts as a mathematical wall: if the weights become enormous, the penalty becomes infinite, forcing the algorithm to stay near the center.
- **Bounded Sublevel Sets:** Thanks to coercivity, if we set a maximum error level (e.g., flooding the bowl up to a certain altitude), the resulting space (the sublevel set) is a closed and compact enclosure. This ensures that the solver's search space is strictly bounded and not infinite.

## Complexity Analysis
The theoretical upper bound for the number of iterations required to guarantee an $\epsilon$-suboptimal solution is described by the formula: $$\mathcal{O}\left(\frac{L^2 D^2}{\epsilon^2}\right)$$
The two geometric parameters:
- **$D$ (Diameter):** The maximum physical size of the search enclosure (the sublevel set). It indicates the maximum horizontal distance the algorithm will have to travel in the worst-case scenario.
- **$L$ (**[[Optimization Difficult#Lipschitz Continuity|Lipschitz]] Constant):** The absolute speed limit for the slope. Mathematically, a function is Lipschitz continuous if it never exceeds a certain vertical change for any given horizontal step: $\vert{}f(x) - f(z)\vert{} \le L\vert{}x - z\vert{}$. This guarantees that the subgradients are bounded ($\Vert{}g\Vert{}_2 \le L$) and forbids the presence of "vertical walls" (infinite slope)    

**Resolving the paradox (From Global to Local):**
The Squared Error contains a parabola. At infinity, the slope of a parabola becomes infinite. Therefore, the ELM objective function is _not_ globally Lipschitz.

However, having proven its coercivity, we know that the algorithm will never go to infinity: it will remain trapped within the compact enclosure of diameter $D$. Within bounded borders, the slope cannot grow to infinity and reaches a maximum local value. Consequently, a finite **local $L$** constant exists, making the theorem theoretically valid and fully applicable.

# References
- [[Standard_Bundle_Methods.pdf]]
- [Bundle methods for stochastic programs](https://svan2016.sciencesconf.org/conference/svan2016/BASLecture26.pdf)
