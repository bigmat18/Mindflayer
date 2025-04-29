**Data time:** 17:31 - 27-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Sampling]]

**Area**: [[Master's degree]]
# Hierarchical Dart Throwing (HDT)

We have a regular grid each cell is the root of a [[Quad-Tree]]. Size of each cell so that it is completely covered by a disk which center is inside the cell.

![[Pasted image 20250427174518.png | 250]]

- **Active cell**: Cell not yet entirely covered by a disk.
- **Active list with index i**: List of active cells at level i.
- **Initial condition**: Active list LO contains all the cells of the grid.

![[Pasted image 20250427174734.png]]

### Cost
**Coverage test**: a secondary uniform grid of cells with size r stores a copy of the point set. By construction, no cell can contain more than 4 points. Therefore, the cost for coverage is constant.

Choice a sample:
1. Keep the sum of areas for each list $a_0, \dots, a_k$
2. Generate a random number in $[0,1]$
3. Find m s.t. $\sum^{m-1}_{i=0} a_i \leq a_{tot}x < \sum^{m}_{i=0} a_i$
4. Pick a random square in the list.

We have a **asyntoptic cost**, it can be determinated by the question "how many cells are created?" authors of this technique said $O(N)$:

*While we cannot provide a rigorous proof, we are convinced that hierarchical dart throwing is O(N) in both space and time on average*

### HDT in 3D space
Surface immersed in a 3D uniform grid where each cell is the root of a [[Quad-Tree|oct-tree]]. The data structure is a direct extensions of HDT except that the cells **is not the** domain but it **contains** the domain (the tingles)

**Basic operation**: once an active cell is chosen, pick a point on the contained surface, use of **pregenerated samples**. 

![[Pasted image 20250427184208.png | 350]]
# References