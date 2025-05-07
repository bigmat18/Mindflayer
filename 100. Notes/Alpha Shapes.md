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
- **Alpha [[Representing real-world surfaces|Complex]]**: Subset of [[Voronoi Decoposition|Delaunay Triangulation]] (if two cells of voronoi are adjacent insert and edge) computed as the dual of the alpha diagram.

L'**alpha complex** is the triangulation built on the alpha diagram.

![[Pasted image 20250507131805.png | 550]]

The **input** is a point cloud:
- In classic voronoi we compute a **[[Voronoi Decoposition]]** and after we do the delaunay triangulation in which we add a edge for voronoi cells adjacent.
- In **alpha shapes** we do alpha diagram, we take the voronoi diagram limited by sphere (we exclude cells to infinity), the following alpha triangulation is a subset of delaunay triangulation and **it's not convex**

Build the **alpha shapes** (for all radius) from the delaunay triangulation is a simple operation, we remove all edge where the circus passing through 3 vertices is bigger than a certain radius

![[Pasted image 20250507132953.png | 400]]
- $\alpha = 0$      $\alpha$-shape is the point set
- $\alpha \to 0$     $\alpha$-shape tends to the convex hull
- A finite number of thresholds $\alpha_0 , \alpha_2 , \dots \alpha_n$ defines all possible shapes (at most $2n² - 5n$)

### Sampling Conditions for Alpha Shapes
Obviously is not sure the results is [[Representing real-world surfaces |manifold]]. This depend to point layout, if we know that the points have a good arrangement probably the final result will be a manifold mesh. The condition to have a good mesh are the following:

Given a smooth manifold M and a sampling S. If it holds that:
1. The intersection of any balls of radius $\alpha$ with M is homeomorphic to a disk 
2. Any balls of radius $\alpha$ centered in the manifold contains at least one point of S.
Then the $\alpha$-shapes of S is homeomorphic to M.

![[Pasted image 20250507140946.png | 350]]


# References