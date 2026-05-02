**Course**: [[Parallel and distributed systems. Paradigms and models]]

**Repository**: https://github.com/bigmat18/distributed-qem-simplification

**Other link**:

# Distributed Mesh Simplification (QEM)

### [[Surface Simplification using Quadratic Error Metric]]
This is a surface simplification algorithm which can rapidly produce high quality approximations of polygonal models. The algorithm uses iterative contractions of vertex pairs to simplify models and maintains surface error approximations using [[Quadratic function|quadric matrices]].
### [[External Memory Management and Simplification of Huge Meshes]]
A paper that describe a data structure called [[Quad-Tree#Oct-tree (3D)|Octree]]-based External Memory Mesh (OEMM). It allow to load dynamically in main memory only the selectred sections and preserving data consistency. This is used to handle memory in a system with multi-thread [[Introduction to OpenMP|OpenMP]] because the data structure is easy to adapt, differently from the most of [[Out-of-Core Simplification]]

# References
- [Surface Simplification Using Quadric Error Metrics](https://www.cs.cmu.edu/~garland/Papers/quadrics.pdf)
- [Scalable Algorithms for Distributed-Memory Adaptive Mesh Refinement](https://charm.cs.illinois.edu/newPapers/12-35/paper.pdf)
- [Mesh Simplification in Parallel](https://www.researchgate.net/profile/Gerhard-Roth/publication/2924678_Mesh_Simplification_In_Parallel/links/00b7d52b9d30e5d88e000000/Mesh-Simplification-In-Parallel.pdf)
- [Distributed Processing of Mesh Simplification](https://diglib.eg.org/server/api/core/bitstreams/57a8a69a-14a2-4995-8f26-efbb8b300c11/content)
- [[Distributed_Processing_Large_Triangle_Meshes.pdf]]
- [External Memory Management and Simplification of Huge Meshes](https://vcgdata.isti.cnr.it/Publications/2003/CRMS03/oemm_tvcg.pdf)