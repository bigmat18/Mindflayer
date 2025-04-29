**Data time:** 01:51 - 26-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Spatial indexing]]

**Area**: [[Master's degree]]
# Binary Space Partition-Tree (BSP) 

This is a [[Hierarchical Indexing Structures]], and in this structure the binary tree is obtained by recursively partitioning the space in two by a hyperplane. Therefore a node always corresponds to a **convex region**. 

![[Pasted image 20250426015717.png | 150]]

![[Pasted image 20250426015753.png | 270]]

![[Pasted image 20250426015814.png | 270]]

![[Pasted image 20250426020349.png | 270]]

![[Pasted image 20250426020407.png | 270]]

This technique has different good property like **all regions are convex**. Unfortunately there are some negative aspects like **query**: is the point p inside a primitive?
- Starting from the root, move to the child associated with the half space containing the point.
- When in a leaf node, check all the primitives.

![[Pasted image 20250426021640.png | 350]]

The cost of search is:
- **Worst**:         O(n)
- **Average**:     O($\log{n}$)

The average case is possible when the **tree in balanced**, and create a balanced tree is not a easy task, in the case where the tree is not balanced the cost can be linear.

Another problem is if the hyperplane is not perfectly symmetric between two primitive (most of the cases) and a search insert point in a region with a primitive but the point is closer to other. To fix this problem, for each steps we need to check if in the near partition there is a closer primitive, in worst case this became linear.

To used this structure we need to do some questions:
- What could go wrong? What happen to split primitives? Can I bound them?
- Where to place the place?

A common strategy is: Primitives are planar faces, use one of the primitives as splitting plane and decompose the rest.

![[Pasted image 20250426023702.png | 350]]

### BSP-Tree Cost
Building a BSP-Tree requires to choose the partition plane. Choose the partition plane that: 
- Gives the best balance? 
- Minimize the number of splits?

The choose depends on the application. In general the cost of BSP-Tree is
$$C(T) = 1 + P(T_L)C(T_L) + P(T_R)C(T_R)$$
Where $P(T_L)$ is probability that $T_L$ is visited given that T has been visited. We can use this formula to choose the splitting primitive. Try to guest the cost: we choose the primitive that minimize
$$1 + |S(T_L)|\alpha + |S(T_R)|\alpha + \beta s$$
- $S_L$ number of primitives in the left subtree
- $s$ number of primitives in split by chosen pimitive

We can have two cases:
- Big $\alpha$, small $\beta$ yield a balanced tree.
- Big $\beta$, small $\alpha$ yield a smaller tree.
# References