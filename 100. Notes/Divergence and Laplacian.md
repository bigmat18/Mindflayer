**Data time:** 15:21 - 29-04-2025

**Status**: #note #master

**Tags:** [[3D Geometry Modelling & Processing]] [[Differential Geometry]]

**Area**: [[Master's degree]]
# Divergence and Laplacian

Here are your notes, revised for better English grammar and flow, while keeping your exact formulas intact. I have expanded on your intuitive explanations to make the concepts easier to visualize and understand.
### Divergence

**Definition:**
Given a vector field $F(F_1, F_2): \mathbb{R}^2 \to \mathbb{R}^2$, the divergence of $F$ is the scalar function $\text{div}: \mathbb{R}^2 \to \mathbb{R}$ defined as:

$$ \text{div} \: F(x,y) = \frac{\partial F_1}{\partial x} + \frac{\partial F_2}{\partial y} $$

**Intuition:**
At a specific point $p_0$, the divergence $\text{div}\: F(p_0)$ measures the extent to which the flow compresses or expands at that location. In simple terms, it calculates how much of the vector field is entering or exiting a single point.

You can think of it by visualizing fluid flow:

- **Positive Divergence ($\text{div} > 0$):** The point acts as a **"source."** More flow is expanding outward and exiting the point than entering it (like a hose spraying water).
    
- **Negative Divergence ($\text{div} < 0$):** The point acts as a **"sink."** The flow is compressing, meaning more is entering the point than exiting (like water going down a drain).
    
- **Zero Divergence ($\text{div} = 0$):** The flow is **"incompressible."** Whatever goes into the point comes right back out at the same rate.
    

---

### Laplacian

**Definition:**
Given a scalar function $F: \mathbb{R}^2 \to \mathbb{R}$, the Laplacian of $F$ is the function $\Delta F: \mathbb{R}^2 \to \mathbb{R}$. It is defined as the **divergence of the gradient** of the function.

$$ \Delta F = \text{div}(\nabla F(x,y)) = \frac{\partial^2 F}{\partial x^2} + \frac{\partial^2 F}{\partial y^2} $$

**Intuition:**
The Laplacian of $F$ at a point $p_0$ measures the extent to which the value of $F$ at $p_0$ differs from the average value of $F$ among its immediate neighbors.

Because it sums up the pure second partial derivatives, it essentially checks the local "concavity" or "curvature" of the function across all directions:

- **Positive Laplacian ($\Delta F > 0$):** The value of the function at $p_0$ is **lower** than the average of its neighbors. This means the point is situated in a local "valley" or "bowl," and the function curves upward around it.
    
- **Negative Laplacian ($\Delta F < 0$):** The value of the function at $p_0$ is **higher** than the average of its neighbors. The point sits on a local "peak" or "hill," and the function curves downward away from it.
    
- **Zero Laplacian ($\Delta F = 0$):** The value at $p_0$ perfectly matches the average of its surroundings. The system is in perfect balance or equilibrium (functions with a Laplacian of zero everywhere are called "harmonic functions").

# References