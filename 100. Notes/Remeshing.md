**Data time:** 21:39 - 13-11-2024

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]]

**Area**: [[Master's degree]]
# Remeshing

Any discretization is an approximation of an ideal shape. For the same abstract shape we can have many different discretizations. No absolute ideal discretization exist. **Remeshing** is concerned with obtaining many different discretization with different property. For example:
- **Closeness/Distance**: how far is my discretization from the intended shapes.
- **Conciseness**: Number of primitive really needed.
- **Shape/Robusteness**: Not all triangles are equals.

### Refinement / Subdivision
We pass from a coarse object to an object more refined. Subdivision defines a smooth curve or surface as the limit of a sequence of successive refinements.

![[Pasted image 20250328141946.png | 350]]

This concept is born from the curve concepts. Simillar to the Bezier Curve and SPLine its start from a small set of control points. This process is determinated from the starting points and tend towards a smooth line. This consepts is extended to 3D space.

Them are used by adjusting the position of a few points of (a) you control the complex shape of a few control point. 

![[Pasted image 20250328143340.png | 350]]


#### Subdivision classification
###### Primal / Dual
How the new mesh is generated:
- **Primal**: Faces split into sub-faces. In this case  we have two type of vertices, the original and the newer.
- **Dual**: New faces for each vertex, edge face. We add a face for each vertex and edge.

![[Pasted image 20250328143926.png | 350]]

###### Approximating / Interpolating
If the original vertices are preserved:
- **Approximating**: Vertexes on the base mash are just control points.
- **Interpolating**: Vertices of the base  mesh stay fixed and you build a surface interpolating them.

![[Pasted image 20250328144709.png | 450]]
We have also triangles and rectangles classification only for primal type.

### [[Doo-Sabin Algorithms]]

### [[Catmull-Clark Algorithms]]

### [[Loop Schema]]

### [[Butterfly]]
### Coarsening / Simplification
You starting discretization is too dense, drop less useful information.
# References