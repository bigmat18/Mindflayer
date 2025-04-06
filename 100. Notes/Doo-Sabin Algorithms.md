**Data time:** 14:53 - 28-03-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Refinement & Subdivision. Remeshing Algorithms]]

**Area**: [[Master's degree]]
# Doo-Sabin Algorithms

This is an algorithms **Dual** and **Approximating**. For each vertex add a face and also for each edge, and it maintain a face for each existing face.

![[Pasted image 20250328145608.png]]
An interesting things is that a face generate by vertex is not a quadrilateral but is a figure with edge equals to number of edges that strike. 

![[Pasted image 20250402180811.png | 300]]

The vertices of new faces come from a weighted sum of previous vertices.
$$V_2 = \frac{1}{n}\cdot \sum{d_j} \:\:\:\:\:\: E_j = \frac{1}{2}(d_1 + d_2) \:\:\: d'_{1, j} = \frac{1}{4}(d_1 + E_j + E_{j-1} + V_j)$$

The $E_i$ are the middle point of vertex of origial mesh. After we calculate the $V$ vertexes taking the average of $d$ points. This is average of average, for this reason the points go inside.
# References