**Data time:** 13:44 - 26-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Spatial indexing]]

**Area**: [[Master's degree]]
# KD-Tree

This is a specialization of [[Binary Space Partition-Tree (BSP)]]. In Kd-tree we have a k dimensions tree. It's a special kind of BSP tree with **axies-aligned bisector planes**. It depends on:
- Choose Axis
- Point on axis where to define the plane

The most important **advantages** with BSP are: 
- test are really fast (to explore the tree)
- lower memory consumption. We must store a coordinated instead of a plane (3 coordinates + offset).

![[Pasted image 20250426135828.png | 300]]

![[Pasted image 20250426135851.png | 300]]

![[Pasted image 20250426135909.png | 300]]

![[Pasted image 20250426135935.png | 300]]

### KD-tree more on cost
Now let's analyse the example of **ray intersection**. 
$$C(T) = 1 + P(T_L)C(T_L) + P(T_R)C(T_R)$$
The cost of a final leaf is roughly the number of primitives (we have to test them). $P(T_L)$ is more interesting:
$$P(T_L) = \frac{|rays\:\:intersecting\:\: T_L|}{|rays \:intersectin\:\:T|}$$
In the ray-tracing, in the formula, the probability for a ray to fall in a section is higher. You can consider rays ad pairs of points over the surface of the cell. Intuitively a ray $(p_1, p_2)$ that hits T hits also $T_L$ iff either $p_1$ or $p_2$ are on $T_L$. With a few assumptions on ray ditrib.
$$P(T_L) = \frac{|surface\:\:area\:\: T_L|}{|surface \:area\:\:T|}$$

![[Pasted image 20250426140842.png | 300]] 

**Input**
- axies-aligned bounding box ("cell")
- list of triangles

**Base Operations**:
- Split a cell using an axies aligned plane ("where?")
- Distribute triangles among the two sets
- Recursive cell

![[Pasted image 20250426141831.png | 600]]

**In the middle**: in this case we have cells with same probability to be intersected, many times we don't found anything and other many times we will go in a cell that has many elements.

**Middle**: in this division we are content based. The left cell has high probability to be selected and many time we must do many useless iteration.

**Cost optimized**: The division is based on number of primitives and the surface of the cells. This is the best solution, because in most cases we enter in a cell with few elements and we know quickly there isn't intersections.

### Range query
A **query** return the primitive inside a given box. It use the following **Algorithm**:
1. Compute intersection between the node and the box
2. If the node is entirely inside the box add all the primitives contained in the node to the result
3. If the node is entirely outside the box return
4. If the node in **partially** inside the box recur to the children

**Cost**: if the leaf nodes contain one primitive and the tree is balanced: $O(n^{1 - 1/d} + k)$, with n = # primitives, d = dimension. Also we have $O(n^{2d})$ possible result.

### Nearest Neighbor
A **query** return the nearest primitive to a given point c. The **Algorithm** is:
1.  Find the nearest neighbor in the leaf containing c.
2. If the sphere intersect the region boundary, check the primitives contained in intersected cells.

![[Pasted image 20250426144711.png | 450]]

The visit of the nearest neighbor is **logarithmic**. 
# References