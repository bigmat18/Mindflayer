---
Data: 2025-12-23T13:15:00
Tags:
  - note
  - "#paper"
  - master
Connection:
  - "[[Distributed Mesh Simplification (QEM)]]"
Area: "[[Master's degree]]"
---
# External Memory Management and Simplification of Huge Meshes

A paper that describe a data structure called [[Quad-Tree#Oct-tree (3D)|Octree]]-based External Memory Mesh (OEMM). It allow to load dynamically in main memory only the selectred sections and preserving data consistency.  This paper in particular deal of [[Out-of-Core Simplification]].

Similar algorithms/techniques with their issues:
- [Out-of-Core Simplification of Large Polygonal Models by Lindstrom](http://www-evasion.imag.fr/Membres/Franck.Hetroy/Teaching/Geo3D/Articles/lindstrom2000.pdf) based on vertex clustering, easy to implement in external memory but simplification accuracy is low, using a [[Vertex Clustering]] techniques.
- [Memory Insensitive Technique for Large Model Simplification by Lindstrom and Silva](https://www.sci.utah.edu/~csilva/papers/vis2001b.pdf): paper that fix the algorithms above but it is two to five times slower.
- [Multiresolution representation for massive meshes by Shaffer and Garlard](https://www.researchgate.net/publication/7985870_A_multiresolution_representation_for_massive_meshes): This is another quality improve of vertex clustering, it replace a regular grid with a BSP trees, even if this method gives an improvement accuracy with respect to standard clustering, the accuracy is lower that that produced with edge-collapse methods.
- [External Memory View-Dependent Simplification by El-Sana](https://cse.engineering.nyu.edu/chiang/EG00-adv.pdf): this approach keep on an external memory an heap ordered according an error  criteria based on edge length or quadratic error metrics (that is not simple). IT has a good computational efficiency if we able to load in memory a large percentage of the data.

##### Mesh Terminology
- A meshes is called **indexed** if all the triangles are encoded by storing a triple of references to their vertices
- Is called **raw** if the triangles are described with a triple of 3D points and sharing of vertices among adjacent triangles is not considered
##### Oct-tree Terminology
We partition the space ion eight sub-region and they are numbered according to their relative coordinates in lexicographic order, which defined ad ordering between octree levels according to DFS visit.

### [[Octree-based External memory mesh]]

### [[Building the OEMM]]

### [[Working with the OEMM]]
# References
- [External Memory Management and Simplification of Huge Meshes](https://vcgdata.isti.cnr.it/Publications/2003/CRMS03/oemm_tvcg.pdf)
- [[oemm_tvcg.pdf|Paper PDF with notes]]