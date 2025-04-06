**Data time:** 11:59 - 03-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Refinement & Subdivision. Remeshing Algorithms]]

**Area**: [[Master's degree]]
# Catmull-Clark Algorithms

This is an algorithm, **polygonal**, **primal** (Every faces will be divided into 4 faces) and **approximating** (the original vertices will be moved). New vertices obtained from existing ones again using appropriate masks. The idea is to subdivide a face to 1 to n surfaces (the number depends to type of mesh).
![[Pasted image 20250403115627.png]]
- **Face:** in the first step we add a vertex, the position is the weighted average of vertexes.
- **Edge:** for the vertex add each edges with use a different weighted average. 

After we had vertexes, we move the origial vertexes with the following patterns based to **valence**: 
- **Valence 3 vertex:** multiplication of coordinate for vertex with 3 faces that strike it.
- **Valence 4 vertex:** multiplication of coordinate for vertex with 4 faces that strike it.
- **Valence 5 vertex:** multiplication of coordinate for vertex with 5 faces that strike it.

![[Pasted image 20250403115851.png | 450]]

The number of quad that affect a vertex is constant during the steps (the grade is maintained). That is different from [[Doo-Sabin Algorithms]] in which to maintain the irregularity use a triangle.
### Algorithm Property
Catmull-Clark has two nice properties:
- Pure quad mesh after one subdivision step.
- The limit surface and its derivate of Catmull-Clark subdivision surfaces can also be evaluated directly, without any recursive refinement. This means that we can do Ray Tracing in a correct way in a subdivision surface without refinement.
# References