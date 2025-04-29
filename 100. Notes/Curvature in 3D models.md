**Data time:** 00:57 - 29-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Differential Geometry]]

**Area**: [[Master's degree]]
# Curvature in 3D models

We can define **curvature** by second derivatives. Define a tangent vector: 
$$t = \cos \phi \frac{x_u}{||x_u||} + \sin \phi \frac{x_v}{||x_v||}$$

![[Pasted image 20250429011835.png | 300]]

This concept work very well in a 2D space, in 3D space we try to reduce the concept of curvature at a 2D domain. To do that consider the plane along n,t and the 2D curve defined on it.

![[Pasted image 20250429012037.png | 300]]

### Curvature in 2D
The curvature of C at P is then defined to be the reciprocal of the radius od osculating circle at point P.

![[Pasted image 20250429012827.png | 250]]

The **osculating circle** of a curve C at given point P is the circle that has the same **tangent** as C at point P as well as the same **curvature**. Just as the tangent line is the line best approximating a curved at a point P, the osculating circle is the best circle that approximates the curve at P.

### Main curvature directions
- For each direction $t$ we define a curvature value k
- Let's consider the two directions $k_1$ and $k_2$ where the curvature values $k_1$ and $k_2$ are **maximum** and **minimum**

There is a **Euler theorem** that said: $k_1$ and $k_2$ are perpendicular and curvature along a direction t making an angle $\Theta$ with $k_1$ is:
$$k_{\theta} = k_1 \cos² \Theta + k_2 \sin²\Theta$$
This theorem has an important handy impact because it say if in any smooth surface the two interesting curvature are perpendicular each other and define a reference frame very useful.

![[Pasted image 20250429014638.png | 230]]

In a smooth surface for each point, if we use this theorem, we have a direction along the way of max curvature, while the other direction will be along the way that not change a lot.

### [[Gaussian Curvature]]

### [[Mean Curvature]]
# References
