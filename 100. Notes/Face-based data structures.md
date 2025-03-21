**Data time:** 16:56 - 13-10-2024

**Status**:  #note #master 

**Tags:** [[3D Geometry Modelling & Processing]] [[Mesh Data Structures]]

**Area**: [[Master's degree]]

# Face-based data structures
### Face set (STL)
In this representation we store the vertex, and the faces. We have:
- **face**: 3 positions

![[Screenshot 2024-10-13 at 20.37.00.png | 300]]
In therm of memory we have, for a triangle mesh, 3 \* 3 \* 4 = 36 bytes per triangle. For the [[Representing real-world surfaces|euler formula]] F is about twice the number of vertices V, this data structure consume 72 bytes/vertex, it's usually call triangle or polygon soup.

Is often not sufficient for most applications
- Connectivity information can't be accessed explicitly 
- Vertices and associated data are replicated as many times as the degree of vertices

### Shared vertex (OBJ, OFF)
In this type of representation the redundancy can be avoided stored an array of vertices and encodes polygons as sets of indices into this array. In this case we have:

- **vertex**: position
- **face**: vertex indices

![[Screenshot 2024-10-13 at 20.39.29.png | 500]]

For the case of triangle mesh, using 32 bits to store vertex coordinates and face indices, this representation requires 12 bytes for each vertex and for each triangle total 12 byte/vertex + 12 byte/face = **36 bytes/vertex** which is only half of the face-set structure.

This data structure reduce the memory usage and, without additional informations it require expensive searches to recover the local adjacency information of a vertex, for this reason **is not sufficiently good for algorithms**.

### Face-based connectivity
This is a standard face-based data structure to optimize the query on triangle mesh, it store the following data:
- **vertex**
	- position
	- 1 face
- **face**
	- 3 vertices
	- 3 face neighbours

![[Screenshot 2024-10-13 at 20.57.29.png | 500]]

With these information one can circulate around a vertex in order to enumerate its one-ring neighbourhood. This representation use **24 bytes/face + 16 bytes/vertex = 64 bytes/vertex**
# References