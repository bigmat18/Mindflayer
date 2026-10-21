---
Data: 2026-10-07T17:27:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computer Vision]]"
Area: "[[Master's Degree.base]]"
---
# Orthographic Projection

[[Perspective projection]] is not the only feasible projection from 3D coordinates of a scene down to the 2D coordinates of the sensor plane. Different camera geometries can lead to other projections. One alternative to perspective projection is **orthographic projection**.

![[Pasted image 20261021162850.png|549]]

In this projection, as the rays are orthogonal to the projection plane, the size of objects is independent of the distance to the camera. The projection equations are generally written as follows:
$$
\begin{aligned}
x &= k X \nonumber \\
y &= k Y
\end{aligned}
\tag{5.5}
$$
- $k$ is a fixed, global scale factor (no dependence on Depth Z at all)

Object size in the image **is independent of depth**: a nearby and a far object of the same physical size project to the same image size. This removes the size-depth ambiguity of perspective projection, but at the cost of **removing depth information from the image entirely**, there is no vanishing point, no perspective foreshortening.

### How to Build an Orthographic Camera?
Can we build a camera that has an orthographic projection? What we need is some way of restricting the light rays so that only perpendicular rays to the plane illuminate the projection plane.

One such camera is the **soda straw camera**, which you can easily build. A set of parallel straws allow parallel light rays to pass from the scene to the projection plane, but extinguish rays passing from all other directions. It works better if the straws are painted black to reduce internal reflections that might capture light rays not parallel to the straws, resulting in a less sharp picture.

![[Pasted image 20261021164316.png|606]]

In the image above: Straw camera example. (a) View through parallel straws. (b) The subject is a hand in sunlight. (c) The resulting image of the straw camera (using smaller straws than (a)). The image projection is orthographic
# References
- https://visionbook.mit.edu/imaging.html#image-formation-by-orthographic-projection