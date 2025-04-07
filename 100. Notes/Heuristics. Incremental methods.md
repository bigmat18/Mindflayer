**Data time:** 15:09 - 07-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Refinement & Subdivision. Remeshing Algorithms]]

**Area**: [[Master's degree]]
# Heuristics. Incremental methods

Fortunately the Heuristics works very well. It are based on **Local Updates Operations**. All of the  methods such that:
1. Simplification proceeds as a sequence of small changes of the mesh (in a greedy way)
2. Each update reduces mesh size and decreases the approximation precision.

###### Vertex removal
![[Pasted image 20250403183650.png]]
In this approach the combinatorial part is more difficult, while the geometric part is very simple.
###### Edge collapse
![[Pasted image 20250403183708.png]]
- Preserve location (one among the 2 vertex)
- New location

Instead in this version the combinatorial part is very simple, while the geometric is more difficult. This is more used i general.
###### Triangle collapse
![[Pasted image 20250403183724.png]]
- Preserve location (one among the 3 vertex)
- New location

The common framework is the following:

![[Pasted image 20250403184947.png | 400]]

##### Mesh optimizations
We can also do a sets of **mesh optimizations**. Simplification based on the iterative execution of: edge collapsing, edge split and edge swap.

![[Pasted image 20250403185145.png | 300]]

Approximation quality evalued with an energy function:
$$E(M) = E_{dist}(M) + E_{rep}(M) + E_{spring}(M)$$
which evaluates geometric **fitness** and repr. **compactness**
- $E_{dist}$: sum of squared distances of the original points from M
- $E_{rep}$: factor proportional to the no of vertex in M
- $E_{spring}$: sum of the edge lenghts
# References