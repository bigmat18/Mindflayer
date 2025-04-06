**Data time:** 12:01 - 03-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Refinement & Subdivision. Remeshing Algorithms]]

**Area**: [[Master's degree]]
# Loop Schema Algorithms

This algorithms is based on **triangular meshes**, and it is **primal** and **approximating**. Edges are spitted and new vertices are reconnected to create new triangles.

![[Pasted image 20250403120421.png | 350]]
### Subdivision

![[Pasted image 20250403120511.png | 550]]

We have two cases to applay weighted sum. 
- For the vertex add on edge (Left figure)
- For the vertex that already exists, in this case it is move (Right figure).

$$E_i = \frac{3}{8} (d_1 + d_{i-1}) + \frac{1}{8}(d_{i-1} + d_{i+1})$$
$$d'_1 = \alpha d_1 + \frac{(1 - \alpha_n)}{n}\sum_{j=2}^{n+1}d_j \:\:\:\:\:\:\: \alpha_n = \frac{3}{8} + \left( \frac{3}{8} + \frac{1}{4} \cos{\frac{2\pi}{n}} \right)^2$$

The new mesh is **continuous**, C0 on extraordinary vertices (valence <4 or >7), C1 elsewhere. 
# References