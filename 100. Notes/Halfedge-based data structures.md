**Data time:** 16:57 - 13-10-2024

**Status**: #note #master

**Tags:** [[3D Geometry Modelling & Processing]] [[Mesh Data Structures]]

**Area**: [[Master's degree]]

# Half-edge based data structures

This representation avoid the case distinctions of edge-based data structures by splitting each edge into two oriented half-edges, we use **counter-clock wise**. For each half-edge we store the following data:
- **Vertex**
	- position
	- 1 outgoing half-edge
- **Half-edge**
	- 1 vertex it points to
	- 1 adjacent face (0 if is boundary)
	- next half-edge of face or boundary
	- previous half-edge in the face
	- opposite half-edge
- **Face**
	- 1 half-edge

![[Screenshot 2024-10-13 at 21.27.45.png | 500]]

Note that the opposite half-edge pointer don't have to be stored if two opposing half-edges are always grouped in pairs and stored in a subsequent array location `halfedges[i]` and `halfedges[i+1]` .

The totally memory consumption in this case is **16 bytes/vertex + 20 bytes/half-edge + 4 bytes/face = 144 bytes/vertex**. With not explicitly storing the previous and opposite half-edge reduces the memory costs to 96 bytes/vertex.
# References