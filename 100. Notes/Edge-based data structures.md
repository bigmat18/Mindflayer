**Data time:** 16:57 - 13-10-2024

**Status**: #note #master

**Tags:** [[3D Geometry Modelling & Processing]] [[Mesh Data Structures]]

**Area**: [[Master's degree]]

# Edge-based data structures

Data structures for general polygon meshes are logically **edge-based**. There two famouse edge based structures:
- **winged-edge**
- **quad-edge**

#### Winged-edge
Where each edge stores reference to its endpoint vertices, incident faces, and to next and previous edge and left and right faces.
- **vertex** 
	- position
	- 1 edge
- **edge**
	- 2 vertices
	- 2 faces
	- 4 edges
- **face**
	- 1 edge

![[Screenshot 2024-10-13 at 21.10.01.png | 500]]

In this representation  in total we have a memory consumption of 16 bytes/vertices + 32 bytes/edge + 4 bytes/face = 120 bytes/vertex.

This representation improve the possibility of the query but have some issues, for example traversing the one-ring still requires case distinctions. To address this issues we can use [[Halfedge-based data structures|halfedge data structures]]
# References