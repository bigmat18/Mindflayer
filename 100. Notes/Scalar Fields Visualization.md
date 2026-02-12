---
Data: 2026-02-12T16:15:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Scalar Fields Visualization

Attributes represented by a scalar function
$$
f : D \to \mathbb{R}
$$
How to encode scalar values? Intuitive idea: Use a perceptual **channel color map**. Arrange continuously a palette of colors and map it to the real line.

![[Pasted image 20260211153644.png | 400]]

### Transfer functions
For volumes, **transfer functions** can map scalars to color and transparency
- Transfer functions are difficult to design
- Semi-automatic tools exist (transfer function design galleries)
- Domain knowledge plays a role

![[Pasted image 20260211153734.png | 550]]

The assignment of color and transparency to density is also known as classification

### Volume Rendering
The process to map a 3D scalar field to **opacity/transparency** and color is called **volume rendering**
There are many techniques but the idea is do **volume ray casting**: each point/cell should contribute; blend contributions by integration.

![[Pasted image 20260211153937.png]]

Visualization of **critical points** (maxima, minima and saddles on surfaces), **isolines** (sets of points on surfaces where the scalar field has the same constant value), and **isosurfaces** (for volumetric data).

![[Pasted image 20260211154005.png | 400]]

Visualization of topological abstractions (e.g., skeletons and topology-preserving graphs)

![[Pasted image 20260211154031.png | 200]]

### Critical Points (continues case)
In the smooth setting, critical points are points of a manifold where the gradient of a smooth scalar function vanishes. Given a smooth function $f: M \to \mathbb{R}$ on a smooth manifold M, a point P
- is a critical point P
$$
\frac{\partial f}{\partial x_1}(p) = 0, \dots \frac{\partial f}{\partial x_{k}}(p) =0
$$
- and a non degenerate critical point if
$$
|H_f (p)| = \bigg | \frac{\partial² f}{\partial x_i \partial x_j}(p) \bigg | \neq 0
$$

![[Pasted image 20260212160905.png]]

A smooth function is called **Morse** if all its critical points are **non-degenerate**. Nice properties:
- Morse functions are a dense subset of all smooth functions
- Critical points are isolated
- Relationship between critical points and the topology of the manifold

### Piece-wise linear (PL) setting
**PL scalar field:** function values provided on vertices and interpolated everywhere else using barycentric coordinates. Does the notion of critical point directly translate from the smooth to the PL setting?

![[Pasted image 20260212161032.png | 500]]

**Drawback**: the gradient of a PL scalar field is piece-wise constant (i.e., constant within each simplex)

We need an alternative definition for critical points. We can use the idea of **connectivity of lower** and **upper link**:
- **The lower link** $LK⁼(v)$ (resp. the **upper link** $LK^+(v)$) of a vertex v relatively to a PL scalar field $f$ is the subset of the link of $v$ such that each its subfaces have a strictly lower (resp. higher) $f$ value than $v$

![[Pasted image 20260212161244.png | 350]]

Vertices can be classified as regular or critical according to the connectivity of upper and lower link:
- **vertex v is regular** if both the lower link $LK^-(v)$ and the upper link $LK^+(v)$ are simply-connected
- **v is a critical** vertex and $f(v)$ a critical isovalue (as opposed to a regular isovalue)

![[Pasted image 20260212161423.png]]

PL scalar field Distinct critical values No degenerate critical point (in practice: **any PL scalar field after perturbation**)

### Level sets
- **Level sets (contours)**: sets of points having the same f-value. 
- **Reeb graphs**: a tool to track the evolution of level sets

![[Pasted image 20260212173754.png | 300]]

Briefly speaking, the **Reeb graph** continuously contract each connected components of level sets to a point, yielding a 1-dim simplicial complex. The topology of level sets changes at critical points

![[Pasted image 20260212173845.png | 500]]

Extrema map to valence-1 vertices, while saddles map to vertices of higher valence

###### Example
![[Pasted image 20260212173914.png | 300]]   ![[Pasted image 20260212173928.png | 250]]

##### Critical points and noise
**Problem:** in practice, critical points may appear in correspondence of slight function ondulations due to noise in the data generation process (acquisition noise, numerical noise in simulation)

To make critical points reliable and useful in practice, one needs a mechanism to classify critical points as either noise or signal. We can use **Persistent Homology**

##### Sub-level sets
The sub-level set of a Morse function f for an isovalue i is the set of points.
![[Pasted image 20260212174130.png | 450]]

### Filtration
Intuition: nested sequence of sub-level sets, according to increasing values of the PL scalar field. As the function **value changes**, **topological features** (connected components in this example) **appear**, **merge**, **disappear**

![[Pasted image 20260212174811.png | 250]]

![[Pasted image 20260212174829.png | 250]]

(This idea can be extended to other topological features: cycles and voidsy)

As the function value changes, topological features (connected components in this example) appear, merge, disappear **Such changes occur at critical points**.

As events occur at critical points, we can associate a **measure of persistence** with pairs of critical points of the **input scalar field**.
- **Persistence** = absolute value of f-value difference of critical points pairs

![[Pasted image 20260212175007.png | 250]]

Intuition: Persistence = lifespan (birth-death interval). Persistence is a measure of the "importance" of critical points. 

Each pair of critical points is represented by a vertical bar, and its persistence is given by the height of the bar.

![[Pasted image 20260212175106.png | 300]]

###### Example: Topological simplification
Find a scalar field with only a subset of critical points. Persistence as the criterion

![[Pasted image 20260212175545.png | 500]]

A hierarchy of Reeb graphs can be obtained by r**epeated persistence-driven removal of pairs of critical points** (hence arcs in the graph)

![[Pasted image 20260212175624.png | 500]]

Segmentations can be obtained by considering the pre-image of each arc in the Reeb graph.

**Persistence diagrams** provide a concise representation of critica point pairs, but they do not provide info on the adjacency relations of pairs on the domains.

![[Pasted image 20260212175654.png | 350]]
# References