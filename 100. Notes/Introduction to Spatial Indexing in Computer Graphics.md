**Data time:** 00:37 - 25-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Spatial indexing]]

**Area**: [[Master's degree]]
# Introduction to Spatial Indexing in Computer Graphics

Let m be a mesh, we can do 3 important questions about this mesh:
- Which is the mesh element closest to a given point p?
- Which are the elements inside a given region on the screen?
- Which elements are intersected by a given ray r?

Let m' be another mesh: 
- Do m and m' intersect? If so, where?

A **spatial search data structures** helps to answer efficiently to these questions.

### Rendering problems
In the rendering this problems becomes fundamental for example for the **path tracing** (aka unbiased ray tracing). From the eye, shoot a ray for each pixel, and find the first surface it encounters.
From this point shoot many other rays and find their intersection, recur until you find either the sky or an emissive surface.

The **core** of the problems is **given a ray find the first primitive it encounters**. You shoot many rays for each hit surface (10~1000). Primitive can easy be O(10⁵) ~ O(10⁹).

![[Pasted image 20250425012753.png | 400]]

### Dynamic/Simulation
Simulating rigid body dynamics requires mainly two takes:
- Computing the position according to current forces
- Computing what are the new forces according the current positions (reaction forces after collision)

Without any spatial search data structure, the solutions to these problems require O(n) time, where n is the numbers of primitives (O(n^2) for the collision detection).

With spatial data structures we can make it (average) almost **constant** or expected logarithmic. Strong  complexity lower bound (worse case log) are possible only for restricted (often not-practical) settings (hard to be proved, reasonable heuristics are the standard).

### [[Non-Hierarchical Indexing Structures]]
It be called also **flat space subdivision**. It would seem trivial, but there are e reasons for them
### [[Hierarchical Indexing Structures]]
It use an approch divide et impera / adaprive subdivision
# References