---
Data: 2025-12-24T13:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Distributed Mesh Simplification (QEM)]]"
Area: "[[Master's degree]]"
---
# Working with the OEMM

There are many operations that we need to performe on this data structure

#### Traversal
We can't load only a leaf at a time, because this don't get the full information. We have differente atomic data access rules:
- **subtree**: load all leaves contained in the subtree plus all the leaf nodes adjacent to the nodes
- **bounding-box**: load the minimal set of leaves such that all the vertices contained in the given bounding-box and all the triangles referencing them are loaded.
#### Loading Leaves
This operation means reconstruct a standard indexed mesh representation from OEMM loaded leaf nodes. This need to create a new mesh with re-indexing of the mesh faces to a new vertex vector composed by the loaded vertices. This operation can be done in **linear-time**.  

We assign 3 flags as follows:
- **not writable**: vertices referenced by triangles outside all $l_i \in S$
- **not readable, not writable**: vertices stored in non-loaded leaves but referenced by triangles in $l_i \in S$ are replaced with dummy vertices.
 
#### Saving Leaves
This operation is needed when a set of leaf nodes has to be written back on secondary memory to make these modifications permanent. To do this we need to convert the current indexed mesh into a OEMM mesh. To ensure correctnedd the following situations must be detected:
- **vertex indices out of range**: if the number of vertices to be saved back in a OEMM leaf is bigger than the original leaf range the leaf range should be expanded. To do this in less cost at OEMM creation time we have distributed the leaf ranges uniformly over 32 bit integer space. In this manner there is plenty of space between any pair.
- **vertex coordinates not contained in the current loaded**: this happens when the coordinates of a modified vertex are not contained in the space corresponding to the loaded OEMM section. To prevent this situation we detect every update which modifies the mesh by moving vertices in regions that are still no loaded.

#### Modify OEMM
###### Node Merging
Every time a leaf is saved back, we firstly check if it can be collapsed with its siblings nodes in the corresponding parent node. If the number of vertices and triangles of the eight siblings is lower than a given threshold we can merge them in a single leaf.
###### Node Splitting
Node splitting is the inverse of the node merging operation, and is has to be performed when the number of element is a leaf is higher that the maximum leaf size.
# References