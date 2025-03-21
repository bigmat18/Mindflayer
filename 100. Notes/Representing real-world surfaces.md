**Data time:** 15:08 - 13-10-2024

**Status**: #note #master 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Representations]]

**Area**: [[Master's degree]]
# Representing real-world surfaces

The real-world surfaces are to much to represent with basic implicit or parametric representation, to avoid this problem surfaces can be represented by **cell complexes**.
#### Cell
A cell is a **convex** (for each couple of points in the object, the segments that join is inside the object) **polytope** (a geometric object extends in many dim, delimited by full faces)
#### Proper face 
A proper face of a cell is a lower dimension convex polytope subset of a cell.

![[Screenshot 2024-10-08 at 13.55.26.png | 250]]
#### Cell complex
A **collection of cells** is a complex iff for each cells $C$ and $C'$ their intersection either is empty or is a common face of both.
![[Screenshot 2024-10-08 at 13.57.22.png | 250]]
- **Order of a cell** is the number of its sides (or vertices)
- **K-complex** is a complex where the maximum of the order of its cells is k.
- **Maximal cell** if it's not a face of another cell
- **Maximal k-complex** is a k-complex where all maximal cells have order k

#### Simplicial complex
A cell complex is a **simplicial complex** when the cells are simplexes (the smaller structure for the space dimensions). A **d-simplex** is the convex hull of d+1 points in.

![[Screenshot 2024-10-13 at 15.25.09.png | 350]]

A collection of simplexes $\sum$ is a simplicial k-complex iff:
- $\forall \sigma_{1} \sigma_{2} \in \sum$ we have $\sigma_{1} \cap \sigma_{2} \neq \emptyset \Rightarrow \sigma_{1} \cup \sigma_{2}$ is a simplex of $\sum$  
- $\forall \sigma \in \sum$ all the faces of $\sigma$ belong to $\sum$ 
- $k$ is the maximum degree of simplexes in $\sigma$ 

![[Screenshot 2024-10-13 at 15.42.00.png | 350]]

#### Face or sub-simplex
A simplex $\sigma'$ is called **face** of another simplex $\sigma$ if it's defined by a subset of the vertices of $\sigma$. Also fi $\sigma \neq \sigma$ if is a **proper face**.

## Topology and Geometry
When we talking about triangle mesh the intended meaning is a **maximal 2-simplicial complex**. It's useful to discrimite between to aspect in mesh:
- **Geometry realisation**: it's about **where** the vertices are actually placed in space.
- **Topological characterisation**: it's about **how** the elements are combinatorially connected.

Give a certain shape we can rappresent it in many different way, topologically different but quite similar from a geometric points of view.
#### Manifoldness
A surface S is a **2-manifold** iff: 
- the neighbourhood of each point is homeomorphic to Euclidean space in two dimension, in other words the neighbourhood of each point is a homeomorphic to a disk (or a semi-disk)

![[Screenshot 2024-10-13 at 15.53.18.png | 300]]
In blue point with a homeomorphic disk and in read point (or set of poits) without a homeomorphic disk.
#### Orientability
A surface is **orientable** it's possible to make a consiste choice for the normal vector.
Two famous shape no-oritable are **Moebius strips** and **klein blottles**.

![[Screenshot 2024-10-13 at 15.55.27.png | 350]]
### Adjacency/Incidency
- Two simplexes $\sigma$ and $\sigma'$ are **incident** if $\sigma$ is a proper face of $\sigma'$ (or viceversa)
- Two k-simplexes $\sigma, \sigma'$  are **m-adjacent** (k > m) i there exists a m-simplex that is a proper face of $\sigma$ and $\sigma'$ 

For example, two triangles sharing an edge are **1-adjacent** instead if we have two triangles that sharing a vertex they are **0-adjacent**.

An intuitive convention to name practically useful topological relations is to use an ordered pair of letters denoting the involved entities.
- **FF**: 1-adjacency (edge adjacent between triangular Faces)
- **EE**: 0-adjacency
- **FE**: proper subface of F with dim 1
- **FV**: proper subface of F with dim 0 (form Faces to Vertices.) 
- **EV**: proper subface of E with dim 0.
- **VF**: F in $\sum$ : V proper subface of F (form a vertex to triangle)
- **VE**: F in $\sum$ : V proper subface of E
- **EF**: F in $\sum$ : E proper subface of F
- **VV**: V' in $\sum$ : it exists and edge E (V, V')

![[Screenshot 2024-10-13 at 16.15.55.png | 200]]
For a two manifold simplicial 2-compex in R3 we have
- FV, FE, FF, EF, EV have bounded degree (are constant it there are no borders)
	- |FV| = 3, |EV| = 2, |FE| = 3
	- |FF| <= 2
	- |EF| <= 2
- VV, VE, VF EE have variable degree but we have some avg estimations:
	- |VV| ~ |VE| ~ |VF| ~ 6
	- |E| ~ 10
	- |F| ~ 2V

### Euler characteristics
The euler characteristic are the following formula
$$\chi = V - E + F$$
- V is number of vertices
- E is number of edges
- F is number of faces

$\chi = 2$ for any simply connected polyhedron. Some examples:

![[Screenshot 2024-10-13 at 16.44.49.png | 400]]

#### Genus
The **genus** of a closed surface, orientable and 2-maniflold is the maximum number of cuts we can make along non intersecting closed curves without splitting the surface in two.

![[Screenshot 2024-10-13 at 16.46.35.png| 300]]

With the genus property we can rewrite the euler characteristics in the following way
$$\chi = 2 - 2g$$
Where **g** represent the genus value. For example:

![[Screenshot 2024-10-13 at 16.48.11.png | 300]]

If we remove a face we have a particular situation:

![[Screenshot 2024-10-13 at 16.49.17.png | 300]]

To resolve these cases we have a different variation of euler characteristics:
$$\chi = 2 - 2g - b$$
Where **b** id the numbers of borders of the surface
# References