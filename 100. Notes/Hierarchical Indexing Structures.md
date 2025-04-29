**Data time:** 01:41 - 26-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Spatial indexing]] 

**Area**: [[Master's degree]]
# Hierarchical Indexing Structures

This types of Inexing Strucures have the goals to resolve all major issues of [[Non-Hierarchical Indexing Structures]]. They use a divide et impera strategies. The space is partitioned in sub-regions recursively/

![[Pasted image 20250426014532.png | 300]]

![[Pasted image 20250426014553.png | 300]]

![[Pasted image 20250426014614.png | 300]]

![[Pasted image 20250426014635.png | 300]]

- The queries correspond to a visit of the tree (in best cases O($\log{n}$))
	- The complexity is sub-linear (logarithmic) in the number pf nodes
	- The memory occupation is linear

- A hierarchical data structures is characterized by:
	- Number of children per node
	- Spatial region corresponding to a node
# References