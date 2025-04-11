**Data time:** 15:11 - 07-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Refinement & Subdivision. Remeshing Algorithms]]

**Area**: [[Master's degree]]
# Greedy Approach (Boundary Error)

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

#### Simplification: Topology Preservation
Edge collapse operation may create non manifoldness

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
#### Efficient Evaluation
Evaluating the error introduced by a collapse efficiently is not trivial. Ideally use [[Remeshing Introduction|Hausdorff]]. This has a problem: at the beginning is easy (few points approximate well H) but at the end it become costly (you need a lot of time to evaluate property)

![[Pasted image 20250407152822.png | 400]]
#### Interpolating Positions (edge collapse)
###### Average Vertex Position
![[Pasted image 20250407153056.png | 350]]

###### Median Vertex Position
![[Pasted image 20250407153131.png | 350]]

###### Quadrics Error Minimization
![[Pasted image 20250407153203.png | 350]]

#### Quadratic Edge Collapse
- Create a plane for each involved vertex, considering their Normals.
- Place the position of the new vertex where it minimize the squared distance to the planes.
- Involves solving a simple linear system.

![[Pasted image 20250407153451.png | 600]]

###### Quadratic Error
Let $n^Tv + d = 0$ be the equation representing a plane. The squared distance of a point $x$ from the plane is
$$D(x) = x(nn^T)x + 2dn^Tx + d^2$$
This distance can be represented as a quadratic:
$$Q = (A,b,c) = (nn^T, dn, d^2) \:\:\:\:\:\:\: Q(x) = xAx + 2b^Tx + x$$
also the sum of the distance of a point from a set of planes is still a quadratic. The error is estimated by providing for each vertex $v$ a quadratic $Q_v$ representing the sum of the all the squared distances from the faces incident in v.

The error of collapsing an edge $e=(v,w)$ can be evaluated as $Q_w(v)$. After the collapse the quadratic of v is updated as follow $Q_v = Q_v + Q_w$

#### Triangle quality
Possibly adding an energy term that penalize bad shaped triangles.

![[Pasted image 20250407154527.png | 250]]

Possibly adding an energy term that tend to balance valence.

 ![[Pasted image 20250407154904.png | 250]]


# References
