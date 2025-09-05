**Data time:** 16:58 - 13-10-2024

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Mesh Data Structures]]

**Area**: [[Master's degree]]

# Direct-Edge bata structures

Is a memory efficient variant of [[Halfedge-based data structures]] that is designed for triangle meshes. Is bases on indices that reference each element in the mesh (vertex, face or halfedge). The indexing implicitly encode some of the connectivity information of the mesh.

Instead of paring opposite half edges, this data structure groups the three halfedges belonging to a common triangle. Let $f$ be the index of a face:
$$
halfedge(f, i) = 3f + i \:\:\:\:i=0,1,2
$$
now let $h$ be the index of a half edge, the index of its adjacent face and its index within that face are simply:
$$
face(h) = h/3 \:\:\:\: face\_index(h) = h \:mod\: 3
$$
the next face index can be computed by $f+1 \:mod\: 3$. The remaining parts of the connectivity have to be stored explicitly, each vertex stores its position and the index to an outgoing halfedge:
- each **vertex** stores its position and index to outgoing halfedge
- each **halfedge** stores the index of its opposite half edge and the index of its vertex.

This lead to a memory consummation of only 16 bytes/vertex + 8 bytes/halfedge = 64 bytes/vertex
# References