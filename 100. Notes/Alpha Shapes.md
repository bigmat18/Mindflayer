**Data time:** 01:40 - 07-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Alpha Shapes

An alpha shapes is a generalization of concept of **Convex Hull**. A Convex hull is the union of all half-spaces that lean on my point cloud. In other words it's the minimal convex solid that I can do around an object.
$$CH(S) = R^d \setminus \bigcup EH(S) $$
Where $EH(S)$ is the halfspace not containing any point is S.

![[Pasted image 20250507014746.png | 200]]

The result has the problem that it is convex, for this reasons it can not approximate not convex shape. For own case we use **Alpha Hull**, where we replace the halfspace with spheres.
$$\alpha H(S) = \mathbb{R}^d \setminus \bigcup EB_{\alpha}(S)$$
Where $EB_{\alpha}(S)$ is the ball with radius $\alpha$ not containing any point S. When the radius tends to infinity we have a sets of plane, when we start to decrease the radius pt's as if the halfplane were curved. 

![[Pasted image 20250507015408.png | 300]]


### Computing Alpha Shapes
To compute alpha shapes we use:
- **Alpha diagram**: [[Voronoi Decoposition |Voronoi Diagram]] restricted to space closest than $\alpha$ one point in S.
- **Alpha [[Representing real-world surfaces|Complex]]**: Subset of Delaunay Triangulation computed as the dual of the alpha diagram

# References