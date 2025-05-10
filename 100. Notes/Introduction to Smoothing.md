**Data time:** 14:15 - 10-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Smoothing]]

**Area**: [[Master's degree]]
# Introduction to Smoothing

The smoothing is used to filtering out the noise form a mesh, we know after [[Range Maps|acquisition]] many noise could be created. Also we want filter out high frequency components for noise removal.

- The smoothing is usually the first steps to work with mesh with different frequency in a different way: this means that if we can separate high with low frequency we can do processing operations like in image processing
![[Pasted image 20250510142151.png | 400]]
- Smooth surface design
![[Pasted image 20250510142420.png | 300]]

- Hole-filling with energy-minimizing patches
![[Pasted image 20250510142359.png | 300]]

- Mesh deformations. Smoothing terms are often needed to obtain nice deformations
![[Pasted image 20250510142333.png | 400]]


The smoothing can see like signals processing, the mesh can be viewed as two-dimensional vector signal defined over a [[Representing real-world surfaces|manifold]]
$$f: M \to \mathbb{R}³ \:\:\:\:\:f(v_i) = x_i$$
where $x_i$ denotes the position of vertex $v_i$ in space and $f$ is extended by linear interpolation to the rest of the mesh. With this representations we can use signal-processing techniques, smoothing at that point view can be viewed as a **filtering process of the signal f**.

### [[Laplacian Smooth]]

# References