---
Data: 2026-10-07T17:26:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computer Vision]]"
Area: "[[Master's Degree.base]]"
---
# Perspective Projection

A [[Pinhole Camera]] projects 3D coordinates in the world to 2D positions on the projection plane of a camera thought the straight line path of each light ray through the pinhole. .

![[Pasted image 20261007173801.png|347]]

The sketch above shows a pinhole camera and the relevant terminology. Instead if we look the image velow it shows the pinhole camera of the image above but **with the box removed**, leaving visible the **projection plane**

![[Pasted image 20261007173941.png|477]]

This image shows also the definition of the three coordinates systems that we will see:
- **World coordinates**: Let the origin of a World Cartesian coordinate system be the camera’s pinhole. The coordinates of 3D position of a point, $P$ , in the world will be $P=(X, Y, Z)$, where the Z axis is perpendicular to the camera’s sensing plane (projection plane).
- **Camera coordinates** (in the **virtual camera plane**): The camera projection plane is behind the pinhole and at a distance $f$ of the origin. Let the coordinates in the camera projection plane, $x$ and $y$, be parallel to the world coordinate axes $X$ and $Y$ but in opposite directions, respectively
- **Image coordinates**: shown in, are typically measured in pixels. Both the camera coordinate system $(x,y)$ and the image coordinate system $(n, m)$ are related by an affine transform as we will discuss later.

If the distance from the sensing plane to the pinhole is then similar triangles gives us the following relations:
$$
\begin{align}
x &= f \frac{X}{Z}
\end{align} \tag{5.3}
$$
$$
\begin{align}
y &= f \frac{Y}{Z}
\end{align} \tag{5.4}
$$
These two equations are called the **perspective projection equations**, where:
- $f$ is the **focal length**
- $Z$ is the **Depth**

![[Pasted image 20261021161345.png|337]]

Coordinates on the virtual plane are not inverted. Image coordinates (𝑥, 𝑦) are proportional to the world coordinates (𝑋, 𝑌 ), scaled by $\frac{f}{Z}$, derived from similar triangles formed by the scene point, the pinhole, and the image point. Under perspective projection, distant objects become smaller, though the inverse scaling by $Z$. 

Due to the choice of coordinate systems, the coordinates in the $x$ virtual camera plane have the coordinate in the opposite direction than the way we usually do for **image coordinates** $(m,n)$, where $m$ indexes the pixel column and $n$ the pixel row in an image. They are **discrete** pixel indices with the origin conventionally at the top-left corner of the image. The relations can be calculated with:
$$
\begin{aligned}
n &= - a x + n_0\\
m &= a y + m_0
\end{aligned}
$$
- $a$ is a constant called **scale factor**
- $(n_0, m_0)$ is the image coordinates of the camera optical axies called **principal point**
- the first minus sign account for the **flip** beetween the camera-coordinates axes and the image-coordinate axes (look the image above with point p)

![[Pasted image 20261021161955.png|309]]
# References
- https://visionbook.mit.edu/imaging.html#image-formation-by-perspective-projection