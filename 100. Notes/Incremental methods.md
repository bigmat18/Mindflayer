**Data time:** 15:09 - 07-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Remeshing. Refinement & Subdivision.]]

**Area**: [[Master's degree]]
# Incremental methods

Incremental algorithm remove one mesh vertex at a time. The candidate is determined based on user-specified criteria. The criteria can be **binary** or **continuous**.
###### Binary
This criteria usually refer to the global approximation tolerance or to other minimum requirements, for example minimum aspect ration of triangles. 
###### Continuous 
This criteria measure the **fairness** of the mesh with respect to t< approximation error or in some other sense such as, for example the isotropic triangles are better than anisotropic ones, or small normal jumps between neighboring triangles are better than large normal jumps.

Fortunately the Heuristics works very well. It are based on **Local Updates Operations**. All of the  methods such that:
1. Simplification proceeds as a sequence of small changes of the mesh (in a greedy way)
2. Each update reduces mesh size and decreases the approximation precision.

### Topological Operations
The major design goal is to keep the operation as simple as possible. This means that we do not want to remove large parts of the original mesh at once but rather want to remove a single vertex at a time.

![[Pasted image 20250403184947.png | 400]]
##### Vertex removal
![[Pasted image 20250403183650.png]]
Delete one vertex plus its adjacent triangles. For a vertex with [[Catmull-Clark Algorithms|valence]] k, this leave a k-sided hole. This hole can be fixed by any triangulation algortihms ([[Delaunay Triangulation]], [[Voronoi Decoposition]]).

Hence, the removal operation decreases:
- the number of vertices one
- the number of edges by three
- the number of triangles by two

In this approach the combinatorial part is more difficult, while the geometric part is very simple.
##### Edge collapse
![[Pasted image 20250403183708.png]]
Collapse edge between two adjacent vertex, both vertex are moved to the same position. If a degenerate triangle has been generate it will be removed. The total operation remove:
- one vertex
- three edge
- two triangle

It also has this property:
- Preserve location (one among the 2 vertex)
- Can be generate new location

Instead in this version the combinatorial part is very simple, while the geometric is more difficult. This is more used i general.
##### Triangle collapse
![[Pasted image 20250403183724.png]]
- Preserve location (one among the 3 vertex)
- New location

The common framework is the following:
##### Halfedge collaps
For two adjacent vertices p and q, p is moved to q's position. This can be consider ad a special case of edge collapsing where ther new vertex position r coincides with q.

This type of collapse has no degrees of freedom. Note that p to q and q to p are treated as independent removal operations.

The big advatage is that for modetate decimation, the global optimization is completely separated from the decimation operator. That makes the design of mesh decimation schemes more **modular**.

![[Pasted image 20250906014912.png]]

###### Link Condition
An important critaria is called **link condition**, which stats  under which conditions an edge collapse preserve the mesh topology. A collapse $(p,q)$ is a valid operation if and only if:
- if both p and q are boundary vertices, then the edge (p,q) has to be a boundary edge
- for all vertices r incident to both p and q there has to be a triangle (p,q,r). In other words, the intersection of the one-rings of p and q consists of vertices opposite the edge (p,q) only

We can see an example in image above where the second point is not true:
![[Pasted image 20250812164543.png | 400]]

##### Vertex contraction
It the above criteria are satisfied, all the above removal operations preserve the mesh consistency and consequentially the topology of the surface.

If a decimation should be able to also simplify the topology of the input model, we have to use **non-Euler removal operators**. The most common is called **vetex contraction**.

In vertex contraction two vertices p and q can be contracted into one new vertex r even if they are not connected by an edge.

This operation reduce the number of vertices by one but keeps the number of triangle constant. This decimation require more flexible data structure. 
### Distance Measures
##### Error Accumulation
The simplest of these techniques is error accumulation. For example if an edge collapse modifies trianlges $t_i$ by shifting one of their corner vertices from p or q to r, the distance of r to $t_i$ is an upper bound for the approximation error introduces in this step.

Error accumulation meas that we store an error value for each triangle and simply add the new error contribution for every decimation step.
##### [[Quadratics Error]]
##### [[Hausdorff Distance]]

### Fairness Criteria
This i the criteria that determinate the order of candidates in the heap. The base criteria is to remove the elements that increase the error least. This criteria is ok but we can use other criteria to optimize result in special cases, examples:
- Prefer triangles with faces that are as close as possible to equilateral.
- If we prefer visually smooth meshes we can use the **maximum** or **average** normal jump between adjacent triangles after the removal as sorting criterion
- Other criteria include **color deviation** or **[[Parametrization Distortion|texture distortion]]**
### Mesh optimizations
We can also do a sets of **mesh optimizations**. Simplification based on the iterative execution of: edge collapsing, edge split and edge swap.

![[Pasted image 20250403185145.png | 300]]

Approximation quality evalued with an energy function:
$$E(M) = E_{dist}(M) + E_{rep}(M) + E_{spring}(M)$$
which evaluates geometric **fitness** and repr. **compactness**
- $E_{dist}$: sum of squared distances of the original points from M
- $E_{rep}$: factor proportional to the no of vertex in M
- $E_{spring}$: sum of the edge lenghts

### Simplification: Topology Preservation
Edge collapse operation may create non [[Representing real-world surfaces|manifoldness]]

![[Pasted image 20250407151525.png | 400]]

 - Let $\sum$ be a 2 simplicial complex without boundary $\sum'$ is obtained by collapsing the edge $e = (ab)$
 - Let $Lk(\alpha)$ be the set of all the faces of the co-faces of $\alpha$ disjoint from $\alpha$.

![[Pasted image 20250407151759.png | 350]]

$Lk(a) \cap Lk(b) = \{x,y\}= Lk(ab)$
![[Pasted image 20250407152031.png | 150]]

$Lk(a) \cap Lk(b) = \{x,y, z, zx\} \neq Lk(ab)$
![[Pasted image 20250407152131.png | 150]]

Mesh with boundary can be managed by considering a dummy vertex $v_d$ and, for each boundary edge e a dummy triangle connecting e with $v_d$. Think it wrapped on the surface of a sphere.

![[Pasted image 20250407152500.png | 150]]
### Efficient Evaluation
Evaluating the error introduced by a collapse efficiently is not trivial. Ideally use [[Hausdorff Distance]]. This has a problem: at the beginning is easy (few points approximate well H) but at the end it become costly (you need a lot of time to evaluate property)

![[Pasted image 20250407152822.png | 400]]

### Interpolating Positions (edge collapse)
###### Average Vertex Position
![[Pasted image 20250407153056.png | 350]]

###### Median Vertex Position
![[Pasted image 20250407153131.png | 350]]

###### [[Quadratics Error|Quadratic Edge Collapse]] Minimization
![[Pasted image 20250407153203.png | 350]]

### Triangle quality
Possibly adding an energy term that penalize bad shaped triangles.

![[Pasted image 20250407154527.png | 250]]

Possibly adding an energy term that tend to balance valence.

 ![[Pasted image 20250407154904.png | 250]]


# References