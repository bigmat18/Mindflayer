---
Data: 2026-05-13T21:56:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# The Proximal Bundle Method (PBM)

To fix the instability of the CP algorithm, it is necessary to **regularize** the Master Problem. This leads to the **Proximal Bundle Method (PBM)**.
1. A **Stability Center** $\overline{x}$: usually corresponding to the best point $x^i$ found so far.
2. A **Stability Parameter** $\mu$: controls "how far from $f_{\mathcal{B}}$ is a good model of f".

![[Pasted image 20260510185458.png | 300]]

##### The Stabilized Master Problem
The original problem is modified by adding a quadratic penalty term. The new Master Problem (which is no longer a linear problem, but a convex quadratic one) becomes:
$$\min \{ f_{\mathcal{B}}(x) + \frac{\mu}{2} ||x - \overline{x}||^2 \}$$

**What is the effect of this formula?**
- It keeps the new point $x^*$ "close" to $\overline{x}$. This enforces stability, acting essentially as a _Trust Region_. The algorithm grafts a "poorman's Hessian" onto the linear model to simulate curvature (like a "poorman's Newton" method).
![[Pasted image 20260510185519.png| 300]]

- **If $\mu$ is too large:** The quadratic term dominates, keeping $x^*$ _too_ close, and the algorithm becomes too slow.
![[Pasted image 20260510185549.png|300]]

- **If $\mu$ is too small:** The quadratic term has little effect, and the algorithm reverts to behaving like the un-stabilized, pure cutting plane algorithm.
![[Pasted image 20260510192302.png|300]]

Dynamically and intelligently managing the center $\overline{x}$ and the parameter $\mu$ is exactly what gives rise to modern **Bundle Methods**.

### The (Proxiaml) Bundle Method
![[Pasted image 20260510185632.png|500]]

As shown in the pseudocode, the PBM manages progress through a two-case logic based on how accurately the model predicts the actual function value (an Armijo-type rule).

1. **Serious Step (SS)**: If the improvement is significant and satisfies the condition $f(x+d^*) - f(x) \le m_1 [f_{\mathcal{B}}(x+d^*) - f(x)]$:
    - The model is considered "good".
    - The point is physically updated: $x \leftarrow x + d^*$.
    - The stability center moves, and the parameter $\mu$ may be reduced.
        
2. **Null Step (NS)**: If the improvement is poor or insufficient:
    - The model is considered "bad" in that specific direction/distance.
    - The point $x$ **remains unchanged** (we stay at the previous stability center).
    - However, the new information obtained by evaluating the rejected point is added to the bundle. This enriches the cutting planes and **improves the model** in that specific area, preparing it for a more informed subsequent attempt.


###  Mathematical Analysis: Why Bundle Methods Work
- **Optimality Condition**: The solution $d^*$ of the stabilized master problem satisfies the inclusion:
$$0 \in \partial [f_{\mathcal{B}}(x+\cdot) + \frac{\mu}{2} ||\cdot||^2](d^*) \implies -\mu d^* \in \partial f_{\mathcal{B}}(x+d^*)$$

- **Convergence**: If the algorithm takes infinite _Serious Steps_, it will descend toward the optimum ($f(x) \rightarrow -\infty$ or $||d^*|| \rightarrow 0$). If serious steps stop and the algorithm executes infinite consecutive _Null Steps_, the model is constantly updated until $||d^*|| \rightarrow 0$. In either case, $||d^*||$ converging to 0 guarantees approaching a global stationary point ($0 \in \partial f(x)$).

- **Practical Efficiency**: Unlike pure subgradient methods, bundle methods exhibit **"fast convergence in the tail"**. This happens because the bundle gradually accumulates enough cutting planes to effectively approximate the "curvature" of the function around the optimum $x_*$    
- **Bundle Compression**: Unlike the original CP algorithm, the bundle size in a PBM can be managed. While a "fat" bundle provides faster convergence in terms of iteration count, bundles can be "compressed" (by removing inactive planes) to save computation time on each individual Master Problem, down to a theoretical limit of just 2 elements per bundle.
# References