**Data time:** 15:04 - 05-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Introduction to Surface Reconstruction

Surface reconstruction is operations to transform from point clouds to tessellated surfaces explicit methods. The **Problem statements** is the following:

Given a Point Cloud $P = \{p_0, \dots, p_n\}, p_i \in \mathbb{R}³$, find the mesh M that it represents.
![[Pasted image 20250506173022.png]]
We can ask our self two questions:
- Q1: It is a very ill posed problem? What does represents means?
A1: ideally, we want to find the surface which sampling produces the input problem.

- Q2: Why do we care about this problem?
A2: Every 3D acquisition device or methods produces a discrete punctual sampling (measures) of surfaces, for example: laser scanning, image based or photogrammetric techniques, computerized axial tomography or simulation data.

###### Laser scanning data source
We can acquire a set of data using laser scanning with a turntable, with static laser scanner (with a specific range of 100, 200 ... meters), or also laser scanning using mobile scanners or airborne LiDAR.

Many of these instruments create a structure from motion (SfM) and Multi-view stereo (MVS)

![[Pasted image 20250506173803.png | 450]]

###### Challenges
The position and normals are generally noisy, the main causes are:
- Sampling inaccuracy
- Scan mis-registration

Moreover, the point samples may not be uniformly distributed:
- Oblique scanning angles
- Laser energy attenuation 

Another important issue is from missing data:
- Material properties, inaccessibly, occlusione can lead to missing data.

### Explicit Methods
Build a tesselation over the point cloud. The points become to vertices of the mesh. It use the vertex of a mesh in a explicit way. In these methods we assume that the point cloud is a right sampling of the surface

![[Pasted image 20250506174716.png]]

We built a triangulation over the point cloud. The points map to vertices of the mesh. This type of techniques are:
- Less robust to noise 
- Require a dense and even sampling
- Generally easier to implement

#### [[Alpha Shapes]]

#### [[Ball Pivoting]]

#### [[Mesh Zippering]]
### Implicit Methods
The steps are:
1. Define the surface implicitly as the zeros of a function $f_p: \mathbb{R}³ \to \mathbb{R}³$
2. Tessellate $\{f_p(p = 0)\}$. It means found the 0 of a function.

![[Pasted image 20250506174730.png]]

Instead of explicit methods, the implicit methods are:
- More robust to noise
- More resilient and uneven sampling. This means that we have a point used to sample a edge of surface, probably it will be ignored.

### Volumetric methods
Tessellate means that the result of this things are a **[[Signed distances field|distance field]]**, i.e. a representations where I have inside a volume a scalar and each of this scalar represents the distance to surface. The sets of points where the distance is 0  represents the surface. 

The standard method to build from a volumetric representations to distance field is the [[Marching cubes]]
# References