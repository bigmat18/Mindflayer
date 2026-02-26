---
Data: 2025-12-15T14:59:00
Tags:
  - note
  - youngling
Connection:
  - "[[Scientific and Large Data Visualisation]]"
Area: "[[Master's degree]]"
---
# Fields (Continues)

**Fields** are a type of dataset in which values associated with a cell contain measurements or calculations from a continuous domain.
- Examples: temperature, speed, force, densities...

![[Pasted image 20251215150015.png | 500]]

Since we have a continuous domain, fields require:
- **Sampling**: how frequently we take the measurements
- **Interpolation**: how to show values in between the sampled points

Field values are stored in grids. Grids have two properties:
- **Geometry**: position of cells in space
- **Topology**: the way cells are connected to adjacent ones

According to grid geometry and grid topology, we have different types of grids: uniform, rectilinear, structured, unstructured

#### Uniform Grids
**Uniform grids**: sampling at regular intervals, no need to explicitly store topology
- also called “raster” (2D grid image, 3D grid volume)
- tessellations of the Euclidean plane/space by square/cube elements. Points are spaced regularly along each direction

![[Pasted image 20251215150202.png | 400]]

#### Rectilinear Grids
**Rectilinear grids**: support non-uniform sampling (efficient storage of information), at the cots of having to store geometric locations
- tessellation by rectangles/rectangular cuboids (not necessarily congruent to each other); regular grids, but spacing of points along axes can vary
- useful to adapt the sampling to the geometry of the data
- connectivity still fixed, yet need to store info for spacing

![[Pasted image 20251215150338.png | 250]]     ![[Pasted image 20251215150346.png]]

#### Structured grids
**Structured grids**: allow curvilinear shape
- same connectivity as uniform/rectilinear grids, but irregular geometry
- points can be placed at an arbitrary coordinate – but no overlapping and self-intersections
- need to store the geometry of each cell

![[Pasted image 20251215150455.png]]

#### Unstructured grids
Unstructured grids: complete flexibility, but need to **store spatial positions** and **topological information** about how the cells connect to each other
- different combinations of cells permitted
- popular choices: triangular and tetrahedral meshes

![[Pasted image 20251215150552.png | 500]]

**Unstructured points** (point clouds):
- no explicit connectivity information
- irregular geometry

![[Pasted image 20251215150635.png | 450]]

According to the values stored in cells, we can distinguish among scalar fields, vector fields, and tensor fields

#### Scalar Fields
**Scalar fields:** each cell contains a single value
- E.g., gray level values in medical imaging

![[Pasted image 20251215150741.png | 550]]

#### Vector Fields
**Vector fields:** each cell contains a vector, represented as direction and modulus
- e.g., velocity

![[Pasted image 20251215150840.png |  550]]

#### Tensor Fields
**Tensor fields:** each cell contains a multi-dimensional array of attributes at each point
- e.g., stress, physical and structural simulations

![[Pasted image 20251215152331.png | 500]]

According to the values stored in cells, we can distinguish among scalar fields, vector fields, and tensor fields
- The values can be **static** or **time-dependent** (this applies to the domain as well, which can deform over time)
- The values can be **deterministic** or **uncertain**
	- e.g., physical models often depend on parameters; one samples the parameter space and runs simulations for each sample
	- this generates a family of scalar fields/vectors/tensors/... (ensamble observations: one for each combination of parameters)
	- major challenge: visualization of uncertain data is a branch of research on its own

**Continuous data** is often found in the form of a **spatial field**, where the cell structure of the field is based on sampling at spatial positions
- The data contain information about both sampled values and the position where they have been acquired/computed
- Related discipline: **[[Scientific Visualization]]**

**Non-spatial data**
- No spatial information provided about the sampling
- The use of space in a visual encoding is chosen by the designer
- Related discipline: Information Visualization
# References