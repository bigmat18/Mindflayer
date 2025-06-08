The project's objective is to develop a high-performance solution, using **CUDA**, for converting a **3D Mesh** (composed by triangles) into a volumetric representation through voxelization. Subsequently, starting from the **Voxelized** representation, a **Signed Distances Field (SDF)** will be calculated, which is a function that, for each point in the volume, returns the distance (with sign) from the mesh surface. This pipeline allows obtaining a volumetric representation rich in geometric information, much more expressive than simple binary voxelization, and finds application in numerous areas.

#### [[Parallel Surface Voxelization]]
**Voxelization** consists of converting the mesh surface into a three-dimensional grid of voxels (cubic cells), marking the voxels that intersect the surface. This step **transforms a continuous geometric representation (mesh) into a discrete one (volume)**, enabling parallel processing and volumetric analysis. Voxelization is the starting point for many 3D processing techniques, as it allows working with structured data that is easily manageable on GPUs.

#### [[Signed Distances Field (SDF)]]
The **Signed Distance Field** is a function that, for each voxel, calculates the minimum distance from the mesh surface, assigning a positive sign (outside the mesh) or negative (inside the mesh). Compared to simple voxelization, the SDF provides a **continuous and differentiable representation of the geometry**.

#### Main Application Areas
- **Computer Graphics and Rendering**: The SDF allows reconstructing smooth surfaces from volumetric data using algorithms like Marching Cubes, and is used for volumetric rendering effects, raymarching, and generating meshes for advanced visualization.

- **Physics Simulation**: In fluid, soft body, or complex collision simulations, the SDF allows for efficient calculation of penetration and physical response between objects, thanks to the continuous distance information.

- **Collision Detection**: The SDF is a useful tool for detecting collisions between 3D objects, as it allows quickly determining if and how much two objects intersect, improving robustness and precision compared to methods based solely on bounding boxes or binary voxels.

- **Deep Learning**: SDFs are used as input for 3D convolutional neural networks and generative models, as they provide a rich and differentiable representation of geometry, facilitating tasks such as reconstruction, segmentation, and generation of 3D shapes.

- **3D Printing and Modeling**: The SDF allows performing offsetting operations (object enlargement/reduction), robust boolean operations, and morphing between shapes, all fundamental operations in preparing models for 3D printing or advanced modeling.

## Project Pipeline

#### Step 1: Transform the Surface into a Grid (Voxelization)
- **Input**: 3D Mesh composed of triangles.
- **Output**: Three-dimensional grid of voxels (binary grid), where each voxel is marked if it intersects the mesh surface.
###### Description
In this phase, the mesh surface is converted into a discrete volumetric representation. The adopted algorithm is the one proposed by Schwarz & Seidel (2010), which allows efficiently and parallely determining which voxels of the 3D grid are traversed by the mesh triangles.

#### Step 2: Calculate the Distance from the Surface for All Voxels
- **Input**: Binary grid resulting from voxelization, where marked voxels represent the surface.
- **Output**: Three-dimensional grid of distance values, where each voxel contains the minimum distance from the mesh surface.
###### Description
Starting from the binary grid, the Euclidean distance to the nearest surface voxel is calculated for each voxel. The algorithm used is the Jump Flooding Algorithm (JFA), which allows iteratively and parallely propagating distance information throughout the grid.

#### Step 3: Determine if Each Voxel is Inside or Outside
- **Input**: Original 3D Mesh and voxel grid (coordinates of voxel centers).
- **Output**: Three-dimensional grid of sign values (+1 for outside, -1 for inside), indicating for each voxel whether it is inside or outside the volume enclosed by the mesh.
###### Description
For each voxel, it is determined whether its center is inside or outside the mesh. The Generalized Winding Numbers method (Jacobson et al., 2013) is used, which calculates a continuous value based on the sum of the solid angles subtended by the mesh triangles with respect to the point of interest.

#### Step 4: Combine Distance and Sign for the Final SDF
- **Input**: Grid of distance values and grid of sign values.
- **Output**: Signed Distance Field (SDF): a three-dimensional grid where each voxel contains the distance from the surface, with a positive sign (outside) or negative (inside).
###### Description
For each voxel, the final SDF value is obtained by multiplying the minimum distance calculated in phase 2 by the sign determined in phase 3. The result is a continuous and differentiable volumetric representation of the mesh geometry.

## State of Art
- **[OpenVDB](https://www.openvdb.org/)**: Excels in managing sparse volumes and SDFs, widely used in the special effects (VFX) industry. It is written in C++ and leverages CPU parallelism (e.g., via OpenMP), but its core routines are not typically implemented directly in CUDA kernels.

- **[CGAL (Computational Geometry Algorithms Library)](https://www.cgal.org/)**: A vast library of computational geometry algorithms. Includes tools for voxelization and distance computation, but they are predominantly CPU-oriented and less focused on massive GPU processing.

- **[VCGlib (Visual Computing Library)](http://vcglib.net/)**: The library underlying software like MeshLab. Contains numerous filters and algorithms for mesh processing, including voxelization and some forms of distance computation, but the primary emphasis is not on end-to-end GPU acceleration from triangle to SDF.

- **Specific Libraries for GPU (e.g., NVIDIA)**: NVIDIA offers SDKs like **[OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix)** (for ray tracing) or **[TensorRT](https://docs.nvidia.com/tensorrt/index.html)** (for AI inference), which can use or require volumetric representations, but they do not provide complete and open libraries for the Voxel+SDF generation pipeline from triangles on GPU.
 


## References
- [Fast Parallel Surface and Solid Voxelization on GPUs" di Michael Schwarz e Hans-Peter Seidel (2010)](https://michael-schwarz.com/research/publ/files/vox-siga10.pdf)
- [Out-of-Core Construction of Sparse Voxel Octrees](https://graphics.cs.kuleuven.be/publications/BLD14OCCSVO)
- [Robust Inside-Outside Segmentation using Generalized Winding Numbers (Jacobson et al., 2013)](https://dl.acm.org/doi/10.1145/2461912.2461916)
- [Jump Flooding in GPU with Applications to Voronoi Diagram and Distance Transform](https://www.comp.nus.edu.sg/~tants/jfa/i3d06.pdf)