**Data time:** 16:57 - 13-10-2024

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Mesh Data Structures]]

**Area**: [[Master's degree]]

# Half-edge based data structures

This representation avoid the case distinctions of edge-based data structures by splitting each edge into two oriented half-edges. For each half-edge we store the following data:
- **Vertex**
	- position
	- 1 half-edge
- **Halfedge**
	- 1 vertex
	- 1 face
	- 1, 2, or 3 half-edges
- **Face**
	- 1 half-edge

![[Screenshot 2024-10-13 at 21.27.45.png | 500]]

Note that the opposite half-edge don't have to be stored if two opposing half-edges are always grouped in pairs and stored in a subsequent array location.

The totally memory consumption in this case is 1**6 bytes/vertex + 20 bytes/half-edge + 4 bytes/face = 144 bytes/vertex**. With not explicitly storing the previus and opposite half-edge reduces the memory costs to 96 bytes/vertex
# References