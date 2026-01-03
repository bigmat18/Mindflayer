---
Data: 2025-12-24T12:18:00
Tags:
  - note
  - youngling
Connection:
  - "[[Distributed Mesh Simplification (QEM)]]"
Area: "[[Master's degree]]"
---
# Building the OEMM
We assume that the input mesh comes as a larger set of raw, not indexed triangles, stored therefore with just 3D coordinates. We consider OEMM construction in the worst-case input.
#### Building Raw OEMM
1. We fix a maximum depth of the octree, then we can all the triangles, for each lead node we count how many triangles are in (a triangle is inside a node at least one vertex is in the node), 
2. When all triangles are virtually assigned to leafs sibling lead nodes are collapsed into the parent node if and only if:
	- the sum of the triangles contained is lower than a user-selected threshold, called `max_triangles` 
	- the resulting merged node has adjacent nodes whose depth in the tree differs from the depth of the current one by no more than three levels

#### Building an Indexed OEMM
We first build an intermediated indexed OEMM where only internal vertices for each leafs are correctly indexed, than the final indexed is build by indexing also the external vertices.
###### Indexing internal vertices
We need to respect the lexicographic order of the leaves. The leaves of the raw OEMM are read ad for each leaf we assign an unique index to each vertex contained in the given leaf. All the vertices that are not contained in the leaf are index with a temporary fake value.
###### Indexing external vertices
Now we compute a global index for all the external vertices of the shared faces.
1. All the internal vertices are read a second time and a global index are assigned.
2. All the lead nodes $l$ which share triangles with a adjacent are loaded
3. For each vertex $v \notin l$ of a shared triangle $t\in l$ we replace the fake index with the correct index assigned to v in the leaf node $l_j$ containing $v$.
# References