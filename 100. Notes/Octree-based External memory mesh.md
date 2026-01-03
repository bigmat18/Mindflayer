---
Data: 2025-12-24T20:38:00
Tags:
  - note
  - youngling
Connection:
  - "[[Distributed Mesh Simplification (QEM)]]"
Area: "[[Master's degree]]"
---
# [[Quad-Tree#Oct-tree (3D)|Octree]]-based External memory mesh

It is based on hierarchy geometric partition of the dataset with no vertex replication and consistent vertex indexing between lead nodes which shared a reference to the same vertex.

A small mesh portion is assigned to each OEMM leaf, based on regular hierarchical decoposition. In main memory us manteined only the octree structure without row data. We load only the data that we need.

An importat feature is that the OEMM maintains a globally indexed representation of the mesh:
- each vertex is uniquely identified by an integer 
- triangles are described and store using just three indices.

###### OEMM leaf node
Each leaf $l$ of the octree stores a pointer to a secondary memory chunk which contains:
- **vertices**: all the vertices contained in the bounding box
- **faces**: for each triangle $t$ partially contained in the bounding box (at leaf one vertex is in the bounding box) is stored in $l$ only if $l$ is the minimal leaf (according to the lexicographic order).

#### Flags
Because we load only a portion of the mesh, we must maintain explicit information on which operations can be performed on the currently loaded or referred mesh elements. For this reasons we use the following flags:
###### Vertex Flags
- **readable** and **writable**: a vertex is
	- readable if is contained in one of the currently loaded leaves
	- writable if all of the faces incident in it are contained in leaves currently loaded
- **modified**: vertex is modified when neither its coordinates of the set of elements incident in it have been modified or processed.

###### Face Flags
- **readable** and **writable**: a triangle is
	- readable if it is contained in one of the currently loaded leaves.
	- writable if all of the vertex-adjacent triangles are readable.
- **modified**: a triangle is modified when its vertex indices have been modified.

# References