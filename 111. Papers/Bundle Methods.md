---
Data: 2026-05-05T00:43:00
Tags:
  - note
  - youngling
  - paper
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Bundle Methods
This section explores **Bundle Methods**, sophisticated algorithms designed to overcome the limitations of simple [[Subgradient Methods]] by utilizing a more comprehensive and accurate model of the objective function.

### The Basic Idea: Building a Better Model
The fundamental idea of the **Cutting Plane** method answers a simple question: _"Want a better direction? Use a better model!"_.

In non-differentiable problems, we do not have second-order information (the [[Hessians]] matrix). The first-order information (the [[(Convex) Non Differentiable Functions#The Concept of a Subgradient|subgradient]]) might seem "crap," but for **[[(Convex) Non Differentiable Functions|convex functions]]**, it has a very powerful property: it is **globally valid**.

If we query an "oracle" at a point $x$, we get the function value $f(x)$ and a subgradient $g \in \partial f(x)$. With this data, we can build a **first-order model** at $x$:

$$I_{x,f(x),g}(z) = f(x) + \langle g, z - x \rangle \le f(z) \quad \forall z \in \mathbb{R}^n$$

This equation represents a hyperplane that _always_ lies below the true function $f(z)$ for any point $z$.

### The Bundle and the Cutting Plane Model
Instead of forgetting previously visited points (as the simple subgradient method does), what if we collect all this information along the way?.

The algorithm stores the data in a **Bundle**, denoted by $\mathcal{B}^i$, which at iteration $i$ contains the history of the explored points:
$$\mathcal{B}^i = \{ (x^h, f^h = f(x^h), g^h \in \partial f(x^h)) \}_{h<i}$$
(Note: cited from the document context where $\mathcal{B}$ represents the bundle )

Using this bundle, we build the **Cutting Plane (CP) model** of the function, which is simply the maximum among all the hyperplanes calculated up to that point:
$$f_{\mathcal{B}}^i(x) = \max \{ I^h(x) = f^h + \langle g^h, x - x^h \rangle : (x^h, f^h, g^h) \in \mathcal{B}^i \} \le f(x) \quad \forall x$$
This model is a convex, piecewise linear function that approximates the real function from below. It is defined as a "$(1+\epsilon)$-order" model.

![[Pasted image 20260510184302.png | 300]]

This work because we know that each hyperplane in $\mathcal{B}$ is below the real function $f(x)$, so if we get the maximum it is the best approximation for the real minimum value of $f(x)$

### The Cutting Plane Algorithm
The algorithm proceeds iteratively by searching for the minimum of the model we just built. The search for this minimum is called the **Master Problem**. 

Even though $f_{\mathcal{B}} \notin C^1$ (it has kinks), finding its minimum is "easy" because it can be solved using Linear Programming, provided that the size of the bundle $\#\mathcal{B}$ is "small".

![[Pasted image 20260510184302.png | 300]]

1. **Solve the Master Problem:** Find the point $x^*$ that minimizes the approximated model and calculate its value $v^*$:    
$$v^* = \min \{ f_{\mathcal{B}}(x) \}$$
this means: which is the lower point in the $f_{\mathcal{B}}$
$$x^* \in \text{argmin} \{ f_{\mathcal{B}}(x) \} \quad \text{with} \quad v^* = f_{\mathcal{B}}(x^*)$$
this instead means, at which coordinates there is this minimun value (we use $\in$ because could be many values with $x^*$ value)

![[Pasted image 20260510184433.png | 300]]

So we have a kinks cup built with all the maximum values in the hyperplanes, and we take the minium part of it, that is own minimum. To do this we need to resolve a Linear Problem.

2. **Oracle Query:** Evaluate the real function at the new candidate point to get a new set of data: $(x^*, f(x^*), g^* \in \partial f(x^*))$.

![[Pasted image 20260510184801.png | 300]]

3. **Stopping Criterion (Check):** If the true function value at $x^*$ is less than or equal to the minimum of the model, we have found the global optimum:
$$f(x^*) \le v^* \Rightarrow x^* \text{ is optimal}$$
    
4. **Model Update:** Otherwise, add the new information to the bundle:
$$\mathcal{B} \leftarrow \mathcal{B} \cup (x^*, f(x^*), g^*)$$
    Now $f_{\mathcal{B}}$ becomes a "better" CP model.
$$
x^* \in \text{argmin} \{ f_{\mathcal{B}}(x) \} \quad \text{with} \quad v^* = f_{\mathcal{B}}(x^*)
$$
![[Pasted image 20260510184856.png | 300]]

**Master Problem as a Linear Program (LP)**: Although $f_{\mathcal{B}}$ is nondifferentiable, finding $x^*$ is "easy" (if the bundle size is manageable) because it can be formulated as a Linear Program (LP):
$$\min \{ v : v \ge f^h + \langle g^h, x-x^h \rangle, \quad (x^h, f^h, g^h) \in \mathcal{B}^i \}$$   
![[Pasted image 20260510184918.png|300]]

A major theoretical advantage of this algorithm is that it provides a **practical and highly reliable stopping criterion**. At each iteration $i$, we can calculate:
- **Lower Bound (Model Value):** $\underline{f}^i = v^{*,i} = f_{\mathcal{B}}^i(x^{*,i}) \le f_*$
- **Upper Bound (Record Value):** $\overline{f}^i = \min \{ f^h : h \le i \} \ge f_*$

The algorithm stops when the difference between the current record and the lower bound is sufficiently small: $\overline{f}^i - \underline{f}^i \le \epsilon$. Under appropriate assumptions, both sequences converge to the true minimum: $\{\overline{f}^i\} \rightarrow f_* \leftarrow \{\underline{f}^i\}$.

### Why the Pure Cutting Plane Algorithm Works Badly
Despite its strong theoretical stopping criterion, the pure CP algorithm often performs poorly in practice.
- **Instability**: The iterates have no **locality property**. Because linear functions have no curvature, the minimum $x^*$ of the model can jump violently from one side of the space to another at each iteration ($||x^{*,i+1} - x^{*,i}||$ can be very large).
- **Lack of Curvature**: You need a massive number of linear functions to approximate the curvature of a quadratic-like function, leading to very slow convergence in the "tail" (final stages).
- **Costly Complexity**: As the iterations increase, the number of constraints in the LP master problem grows, making each iteration increasingly expensive (up to $O((1/\epsilon)^{n/2})$ in some cases).

![[Pasted image 20260510185015.png|300]]

![[Pasted image 20260510185028.png | 300]]

### [[Trust-Region Stabilization]]
### [[The Proximal Bundle Method (PBM)]]
### [[Level Stabilization (PLBM)]]
### [[Center-Based Approaches]]


# References
- [[Standard_Bundle_Methods.pdf]]