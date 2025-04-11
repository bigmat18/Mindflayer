**Data time:** 11:55 - 10-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]]

**Area**: [[Master's degree]]
# Introduction to Parametrization

##### Texture mapping
Parametrization is a classing problem in 3D modelling and it's basic concepts for **texure mapping**. In many areas this is an advanced artistic skill using a manual UV mapping.

##### Remeshing
Also parametrization is useful in many [[Remeshing Introduction|remeshing]] context. This because there is global approach to remeshing that use parametrization, if I obtain a parametrization in a specific domain, a grid on this domain applaied on the mesh it's a very good base to do a remeshing.

By **global approch** we mean to use the grid instead of mesh.
![[Pasted image 20250411161153.png | 400]]
##### Analysis 3D surface
We can use 2D surface created by parametrization to analysis 3D (2D is easier than 3D). For example there are many strategy for image segmentation, with parametrization we can apply image technique on a mesh surface.

![[Pasted image 20250411162033.png | 250]]
##### What we need for good parametrization
To parameterize a surface we need two basic elements:
- a strategy to flatten a 3D surface on 2D domain, introducing a few distortion as possibile

![[Pasted image 20250411162624.png | 500]]
- a strategy to introduce cuts. If was possible to generate an infinite number of cuts any mesh can be parametrizated a triangle at the time. This solution is not possible because every cuts introduce a complexity problem.

![[Pasted image 20250411162706.png | 500]]

## Flattening a Surface
We define a **surface** like $S \subset \mathbb{R}^3$, and a parameter domain $\Omega \subset \mathbb{R}^2$. Mapping using $f: \Omega \to S$ and $f^{-1}: S \to \Omega$ In research with intend $f$ like a function from 2D to 3D (inverse of real context), one of the reasons much of literature was born to describe a complicated surface with a parametrization  for example a torus can be describe with a [[Parametric representations | parametric function]] 

![[Pasted image 20250410120250.png | 500]]

##### Ortho Projection Example 

![[Pasted image 20250410120529.png | 500]]
$$S = \{(x,y,z) \in \mathbb{R}^3 > x^2 + y^2 + z^2 = 1, z \geq 0\} \:\:\:\:\:\: \Omega = \{(u,v) \in \mathbb{R}^2 : u^2 + v^2 \leq 1\}$$
$$f^{-1}(x,y,z) = (x,y) \:\:\:\:\:\:\: f(x,y) = (x,y, \sqrt{1 - u^2 - v^2})$$
##### Cylindrical Coords Example

![[Pasted image 20250410121428.png | 500]]
$$S = \{(x,y,z) \in \mathbb{R}^3 : x^2 + y^2 = 1, z \in [0,1]\} \:\:\:\:\:\: \Omega = \{(\phi, h) \in \mathbb{R}^2 : \phi \in [0, 2\pi), h \in [0, 1]\} \:\:\: f(\phi, h) = (\sin \phi, \cos\phi, h)$$
In this example we represent in polar coordinates every points of cylindrical surface.

# References
