**Data time:** 01:24 - 26-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Spatial indexing]]

**Area**: [[Master's degree]]
# Spatial Hashing

This is the first solution for the main problems of [[Uniform Grid]] system. It's very similar to a uniform grid, except that only non empty cells are allocated.

![[Pasted image 20250426012625.png | 500]]

This solution is more optimised for memory but introduce a slight cost for the search (introduced by the hash function).
- **Cost**: Same as UG, except that in worst case the access to a cell is O(#cells) because of collisions.
- **Memory occupation**
	- **Worst**: all volumetric cells are used
	- **Average**: only a few surface **intersecting** cells are allocated.

##### Pros
- Fast query if **goof hashing** is done
- Easy to implement
- Less memory consuming
##### Cons
- Performance **very** sensitive to distribution of primitives


# References