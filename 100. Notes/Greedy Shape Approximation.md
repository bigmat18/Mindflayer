**Data time:** 17:12 - 11-08-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Mesh Simplification and Approximation]]

**Area**: [[Master's degree]]
# Greedy Shape Approximation

A greedy algorithm to compute an [[Variational Shape Approximation|approximate minimum equation]] is proposed. Its main advantages are:
- The algorithm generates multiresolution hierarchy of shape approximations
- the output is guaranteed to be free of fold-overs and degenerated faces

This algorithm requires a robust computation of [[Delaunay Triangulation|Delaunay Triangulation]] 

```
for each region:
	1. evaluate quality after simulated operation
	2. put the operation in the heap (quality, reagion)
```

Reapet the following operations until no further reduction possibile:
1. pick best operation form the heap
2. if introduces error $< \epsilon$
	- Execute the operation
	- **Update heap**

### Algorithm
##### Setup
Otherwise [[Variational Shape Approximation]] in additional to $R, P$ sets the algorithm maintains a set of polygonal faces $F = \{f_1, \dots, f_k\}$. Each face $f_i$ can be an arbitrary connected polygon, ie, it has on outer boundary and possibly a number of inner boundary around interior holes.

We initialize R, P and F as follows:
- $R_i = \{t_i\}$, ie, each triangles makes up a region on its own.
- $P_i = (x_i,n_i)$ where $x_i$ is an arbitrary point on $t_i$ and $n_i$ is $t_i$'s normal.
- $f_i = t_i$ in particular the projection of $f_i$ onto $P_i$ is injective.

##### Algorithm Invariant
To guarantee a valid spahe approximation we follow a invatiant at alla times during the run of the algorithm:
- **Injectivity constraint**: the projection of $f_i$ onto $P_i$ is injective

Due to the contraint, we are able to extract a valid triangle mesh at alla times during the run of the algorithm.

##### Greedy Optimization
The greedy optimized in a loop that stop when:
- A predefined maximum error is reached
- Or a predefined number of regions is reached

In each iteration:
1. Select two regions $R_i$ and $R_j$ 
2. Merges them into a new region $R' = R_i \cup R_j$ 
3. Compute a new proxy $P' = (x', n')$ as area-weighted average of $P_i$ and $P_j$
$$
n' = \frac{a_in_i + a_kn_j}{||a_in_i + a_jn_j||} \:\:\:\: x' = \frac{a_ix_i + a_jx_j}{a_i + a_j}
$$
	where $a_i = area(R_i)$
4. Check for valence-2 vertices
	- If it finds an interior valence-2 vertex it is immediately removed

##### Merge Priorities
For each adjacent pair $R_i$ and $R_j$ of regions, we could compute the [[Variational Shape Approximation|shape measure E(R', P')]] and order the region pairs by increasing shape error.

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
- [Optimization Techniques for Approximation with Subdivision Surfaces by Marinov and Kibbelt](https://diglib.eg.org/server/api/core/bitstreams/b4d44a8e-8d61-40e4-be0c-dd6cd1a862fd/content)