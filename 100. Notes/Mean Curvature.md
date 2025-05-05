**Data time:** 14:37 - 29-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Differential Geometry]]

**Area**: [[Master's degree]]
# Mean Curvature

Is a concept related to [[Gaussian Curvature]], but it's more simple. It's defined by the mean of two curvatures, like this:
$$H = \frac{(k_1 + k_2)}{2}$$
Measure the **divergence** o normal in a local neighborhood of the surface. 
- if all vector are parallel we haven't divergence
- If I have a source point we will have divergence 

This concept can be also say like: image a vector field represents water flow:
- If $div_s$ is a **positive** number, then water **is flowing out** of the point.
- if $div_s$ is a **negative** number, then water is **flowing into** the point.

![[Pasted image 20250429145335.png | 500]]

The **divergence** $div_{S}$ is a operator that measures a vector field's tendency to originate from or converge upon a given point.

### Minimal surface and minimal area surfaces
A surface is **minimal** iff $H = 0$ everywhere. In this case the normals are parallel, and we can't move vertices further to minimize, the surface so is minimal. 

Another important thing is all surfaces of minimal Area (subject to boundary constraints) have $H = 0$ (not always true the opposite)

![[Pasted image 20250429151308.png | 600]]


### Mean curvature on triangle mesh
We use the following formula based on [[Gradiant, Divergence and Laplacian|Cotangent formula]]:
$$H(p) := \frac{1}{2A(v)} \sum (\cot \alpha_i + \cot \beta_i)||p_ - p_i||$$
Where $\alpha_i$ and $\beta_i$ are the two angles opposite to the edge in the two triangles having the edge $e_{ij}$ in common, A is the sum of the areas of the triangles.

![[Pasted image 20250429164904.png | 400]]

### Screen space mean curvature
The mean curvature is used un Computer Graphics to view a mesh with the edge bold and improve the readability of the model. It's also called **Cavity Shading**.

![[Pasted image 20250429173209.png | 300]]
# References