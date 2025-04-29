**Data time:** 01:36 - 25-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Spatial indexing]]

**Area**: [[Master's degree]]
# Uniform Grid

This is a **non-hierarchical** indexing structures where the space including the object in partitioned in **cubic cells**; each cell contains references to "primitives" (ess. triagles).
##### Construction 
Primitives are assigned to the cell containing their feature point (ess. barycenter or one of their vertices). All the cells spanned by each primitive.

![[Pasted image 20250426010835.png | 200]]

**Regular grids access by position is trivial**: 
if you want to know if something is at (x,y,z) just use integer division.
##### Closest element (to point p)
- Start from the cell containing p
- check for primitives inside growing spheres centered at p.
- At each step the ray increases to the border of visited cells

![[Pasted image 20250426010915.png | 200]]

For this case we have this **cost**:
- **Worst**:          O(#cells + n)
- **Average**:      O(1)

##### Intersection with a ray
1. FInd all the cells intersected by the ray
2. For each intersected cell, test the intersection with the primitives referred in the cells
3. Avoid multiple testing by flagging primitives that have been tested (mailboxing)

![[Pasted image 20250426011419.png | 250]]

In this case we have the following **cost**:
- **Worst**:          O(#cells + n)
- **Average**:      O($\sqrt[d]{\#cells} + \sqrt[d]{n}$) 

**Memory occupation**: O(#cells + n)
##### Pro
- Easy to implement
- Fast query
##### Cons
- Memory consuming. In the basic case the major of the grid will be empty.
- Performance very sensitive to distribution of the primitives. This because there is many cache miss caused by the high number of empty cell. 

### Cells size
The choice of the cells size is very important for the final result. If the cells are too small there will be too many empty cells, if the cells are too large inside each cells there will be too many elements.

![[Pasted image 20250426014025.png | 500]]
# References