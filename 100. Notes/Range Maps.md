**Data time:** 02:38 - 09-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Range Maps

Not only point cloud: the **Range Maps** or structured point cloud. [[Introduction to Surface Reconstruction|3D scanners]] produce a number of dense structured height fields, that is, a regular $(X,Y)$ grid of points with a distance Z value. There are called **range maps**. They are trivial to triangulate use the regularity of sampling.

![[Pasted image 20250507154505.png | 500]]


In sintesi a range maps is a grid with, for each block of the grid a value that express the depth, the total acquisition is a n set of 3D coordinates expressed on the camera coordinates system

One of the main question is how to merge different range maps? It's very difficult for the different noise and local deformations of surface.
### Scanning pipeline for range maps
![[Pasted image 20250509163618.png ]]
1. **Scanning**: for each view points we have a partial range map, we need to move each in a single coordinate system.
![[Pasted image 20250509163736.png | 400]]

2. **Registration**: this is the process to move each range map in a single coordinate system. It may be spitted: 
	- **Cores registration**: more approximated, it is called also [[RANSAC Random Sample Consensus|raf alignment]]
	- **Fine registration**: registration with more precision, like with [[ICP Iterative Closest Point]]

![[Pasted image 20250509164125.png | 300]]

3. **Stitching/[[Introduction to Surface Reconstruction|Reconstruction]]**

![[Pasted image 20250509164156.png | 150]]

4. **Post-process**: all operations like: [[Remeshing Introduction|remeshing]], filtering ecc..

# References