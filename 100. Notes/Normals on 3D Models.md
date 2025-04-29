**Data time:** 00:36 - 29-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Differential Geometry]]

**Area**: [[Master's degree]]
# Normal on 3D Models

### Normal

Let's consider a 2 manifold surface S in $\mathbb{R}³$. Suppose to have a mapping $\mathbb{R}² \to \mathbb{R}³$ 
$$S(u,v) \to \mathbb{R}³$$

Then we can define the normal for each point of the surface as: 
$$n = (x_u \times x_v) / ||x_u \times x_v||$$
Where $x_u$ ans $x_v$ are vectors on tangent space.

![[Pasted image 20250429004159.png | 250]]

This definition is easy in a parametric surface, but in a discrete surface we can't use this definition.
### Normal on triangle meshes
Computed per-vertex and interpolated over the faces. Usually we consider the tangent plane ad the average among the planes containing all the faces incident on the vertex. 
$$n_v = \frac{1}{\#N(v)}\sum_{f \in N(v)}n_f \:\:\:\:\: N(v) = \{f : f \: \:coface\:\:of\:\:v\}$$

![[Pasted image 20250429004754.png | 230]]

This work well for a **good tessellation**, because if I do only the average of all normals can be wrong in some situations, indeed small triangles may change the result dramatically.

![[Pasted image 20250429005126.png | 250]]

Weighting by area, angle, edge len, helps to fix this issue. If you get normal as cross product of adj edges, if you leave it un-normalized its length is twice the area of the triangle, we can get the area weighting for free.
# References