**Data time:** 16:13 - 17-08-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Remeshing. Mesh Simplification and Approximation]]

**Area**: [[Master's degree]]
# Out-of-Core Simplification
Mesh simplification is often applied on verty large sets that are complex to fit in main memory. Many out-of-core algorithms have benne proposed that allow an efficient decimantion of polygonal meshes.

#### [Out-of-Core Simplification of Large Polygonal Models by Lindstrom](http://www-evasion.imag.fr/Membres/Franck.Hetroy/Teaching/Geo3D/Articles/lindstrom2000.pdf) 
It is based on **[[Vertex Clustering]]** combined with **[[Quadratics Error]]**. This approch need only as single pass over the mesh data to build incrementally a in-core representation of the simplified mesh. 
	
This approach use also a dynamic **hash table** for fast localization, and quadrics associations with a cluster. 

The final simplified mesh is then produced by computing a representative from the pre-cluster quadrics and the corresponding connectity information as descrived above. 
#### [Memory Insensitive Technique for Large Model Simplification by Lindstrom and Silva](https://www.sci.utah.edu/~csilva/papers/vis2001b.pdf)
This paper improve the initial work of Lindstrom by removing the requirement for the output model to fit into main memory using a multi-pass approach.
- This method require only a constant amount of memory
- The memory required is independent from the size of the input and output

#### [[Stream Algorithm for the Decimation of Massive Meshes by Wu and Kobbelt]]

#### [Large mesh simplification using processing sequences by Isenburg](https://www.researchgate.net/publication/4046820_Large_mesh_simplification_using_processing_sequences)
Introduced **mesh processing sequences** which represent a mesh as as fixed interleaved sequence of indexed vertices and triangles. Processing sequence can be used to improve [[Stream Algorithm for the Decimation of Massive Meshes by Wu and Kobbelt]] algorithm.

#### [Multiresolution representation for massive meshes by Shaffer and Garlard](https://www.researchgate.net/publication/7985870_A_multiresolution_representation_for_massive_meshes) 
A schema that combine out-of-core vertex clustering step with an in-core [[Quadratics Error|iterative decimation]] step. The idea is that the ordering of edge collapses is only relevant for very coarse approximations, thus, the decimation process can be simplified by combining many edge collapse operations into single [[Vertex Clustering]] operations to obtain an intermediate mesh.
# References