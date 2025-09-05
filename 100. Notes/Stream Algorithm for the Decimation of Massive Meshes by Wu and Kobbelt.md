**Data time:** 16:45 - 17-08-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Remeshing. Mesh Simplification and Approximation]]

**Area**: [[Master's degree]]
# Stream Algorithm for the Decimation of Massive Meshes

### Ideal Stream Algorithm for Decimation
We consider a input [[Channels in Message Passing|stream]] of triangle, each triangle is given by its three vertices with three coordinates for each. Define $N_{max}$ the maximum number of triangles that fit into the in-core triangle buffer.

The stream perform three different operations that affect the filling level $N_{current} \leq N_{max}$ of the buffer.
###### READ(k)
Takes the next $k$ triangles from the input stream and insert them into the current in-core portion of the mesh $N_{current} \leftarrow N_{current} + k$  
###### DECIMATE(k)
Performs $k$ edge collapse operation on the in-core portion of the mesh according to the multiple choice optimization strategy. Each edge collapse removes two triangles from the mesh. $N_{current} \leftarrow N_{current} = 2k$
###### WRITE(k)
Removes $k$ triangles from the in-core portion and writes the into the output stream. $N_{current} \leftarrow N_{curr} \leftarrow k$

These operations must be applied in arbitrary order:
-  Hard restriction is $N_{current} = N_{max}$ we have to apply DECIMATION or WRITE before reading.
- Weak restriction are 
	- the filling leve should be kept as high as possibile 
	- the number of DECIMATION and WRITE should be balance to achieve the target resolution
### Real Stream Algorithm for Decimation

# References
- [A Stream Algorithm for the Decimation of Massive Meshes by Wu and Kobbelt](https://www.graphics.rwth-aachen.de/media/papers/streamdeci1.pdf)
