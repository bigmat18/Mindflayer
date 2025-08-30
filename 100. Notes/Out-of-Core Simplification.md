**Data time:** 16:13 - 17-08-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Mesh Simplification and Approximation]]

**Area**: [[Master's degree]]
# Out-of-Core Simplification

Mesh simplification is often applied on verty large sets that are complex to fit in main memory. Many out-of-core algorithms have benne proposed that allow an efficient decimantion of polygonal meshes.

### [Out-of-Core Simplification of Large Polygonal Models by Lindstrom](http://www-evasion.imag.fr/Membres/Franck.Hetroy/Teaching/Geo3D/Articles/lindstrom2000.pdf) 
It is based on **[[Vertex Clustering]]** combined with **[[Quadratics Error]]**. This approch need only as single pass over the mesh data to build incrementally a in-core representation of the simplified mesh. 

This approach use also a dynamic **hash table** for fast localization, and quadrics associations with a cluster. 

The final simplified mesh is then produced by computing a representative from the pre-cluster quadrics and the corresponding connectity information as descrived above. 
### [A Memory Insensitive Technique for Large Model Simplification by Lindstrom and Silva](https://www.sci.utah.edu/~csilva/papers/vis2001b.pdf)
This paper improive the initial work of Lindstrom by remobing the requirement for the output model to fit into main memory using a multi-pass approach.
- This method require only a constant amount of memory
- The memory required is independent from the size of the input and output

### [[A Stream Algorithm for the Decimation of Massive Meshes by Wu and Kobbelt]]

### [Large mesh simplification using processing sequences by Isenburg](https://www.researchgate.net/publication/4046820_Large_mesh_simplification_using_processing_sequences)

### [Multiresolution representation for massive meshes by Shaffer and Garlard](https://www.researchgate.net/publication/7985870_A_multiresolution_representation_for_massive_meshes) 
# References