**Data time:** 15:38 - 13-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]] [[Parametrization Techniques]]

**Area**: [[Master's degree]]
# Mass-Spring Parametrization

We can image the problem like a Mass and Spring problem. These types of problem we try to model dynamic structures like graph with a mass on nodes and springs on edges. With this model we have
- Position of vertices $p_0, \dots, p_n$
- UV positions of vertices $u_0, \dots, u_n$
### Relaxation process
We image that positions in the space of points (consider to semplicity the 2D space), and consider the length of edge equals to the length of spring at rest. 
1. Map the extreames vertices on two points

![[Pasted image 20250413155650.png | 250]]
2. The positions of third points is a value based on springs. This means if first spring is more strong the position $u_j$ will be moved toward the first vertex.

![[Pasted image 20250413155730.png | 250]]
### Energy Minimization
- Energy of spring between $p_i$ and $p_j$:        $\frac{1}{2}D_{ij}s_{ij}²$
- Spring constant (stiffness):                      $D_{ij} > 0$
- Spring length (in parametric space):        $s_{ij} = ||u_i - u_j||$
- Total energy:                                              $E = \sum_{(i,j)\in \upvarepsilon} \frac{1}{2}D_{i,j}||u_i - u_j||²$
- Partial derivate:                                          $\frac{\partial E}{\partial u_i} = \sum_{j \in N_i} D_{ij} (u_i - u_j)$

In the 3D cases er image that we have already parametrizated the vertices around the vertex that I want compute. We must found the energy of $p_i$ that minimize the energy of the system.

![[Pasted image 20250413163428.png | 250]]

$u_i$ is expressed ad a **convex combination** of its neighbours $u_j$
$$u_i = \sum_{j \in N_j} \lambda_{ij} u_j$$
with weights are $\lambda_{ij} = D_{ij} / \sum_{k \in N_i} D_{ik}$. Lead to Linear System.

![[Pasted image 20250413163956.png | 250]]
###### Uniform Spring Constants
![[Pasted image 20250413164111.png]]
###### Proportional to 3D distance
![[Pasted image 20250413164127.png]]
#### No Linear Reproduction
With this approach we can't reproduce linear situations. If we have a planar mesh it is distorted. 

![[Pasted image 20250413164402.png | 500]]

For this reasons there are different approach to found weights to express a vertex with the sum of neighbours. Suppose S to be is planar.
1. Specify weights $\lambda_{ij}$ such that:     $p_i = \sum_{j \in N_i} \lambda_{ij}p_j$
2. Then solving:                                $u_i = \sum_{j \in N_i} \lambda_{ij}u_j$
3. Reproduces S

###### Watchpress coordinates
$$w_{ij} = \frac{\cot \alpha_{ji} + \cot{\beta_{ij}}}{r_{ij}²}$$###### [[Harmonic Parametrization|Discrete harmonic coordinates]]
$$w_{ij} = \cot{\gamma_{ij}} + \cot{\gamma_{ji}}$$
This maintain the proportions during parametrizations
###### Mean value coordinates
$$w_{ij} = \frac{\tan{\frac{\alpha_{ij}}{2}} + \tan{\frac{\beta_{ji}}{2}}}{r_{ij}}$$
![[Pasted image 20250413170420.png | 200]]


# References