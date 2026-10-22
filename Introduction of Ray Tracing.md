---
Data: 2026-10-20T18:35:00
Tags:
  - note
  - youngling
Connection:
  - "[[Ray Tracing Walktrough]]"
Area:
---
****# Introduction of Ray Tracing

When somebody says “ray tracing” it could mean many things. What I am going to describe is technically a path tracer, and a fairly general one.
- The algorithm is based to analyze the behavior of a ray inside the scene. 
- We will create a ray for each pixel

A **ray** is essentially a function with this form $P(t) = A + tb$. Here:
- $P$ is a 3D position along a line in 3D
- $A$ is the ray origin
- $b$ is the ray direction.
- The parameter $t$ is a real number

Plug in a differente $t$ and $P(t)$ moves the point along the ray, moves the point along the ray. Add in negative $t$ values and you can go anywhere on the 3D line.

![[Pasted image 20261020190647.png|385]]

### Sending Rays Into a Scene
Now we are ready to turn the corner and make a ray tracer. At its core, a ray tracer sends rays through pixels and computes the color seen in the direction of those rays. The involved steps are
1. Calculate the ray from the “eye” through the pixel,
2. Determine which objects the ray intersects, and
3. Compute a color for the closest intersection point.

> [!DEFINITION]
> **Aspect-Ratio**
> A square image has a 1∶1 aspect ratio, because its width is the same as its height. Since we want a non-square image, we'll choose 16∶9 because it's so common. A 16∶9 aspect ratio means that the ratio of image width to image height is 16∶9. Put another way, given an image with a 16∶9 aspect ratio,
> $$
> \text{screen\_width}/\text{screen\_height} = 16 / 9 = 1.7778
> $$
> The image's aspect ratio can be determined from the ratio of its width to its height. However, since we have a given aspect ratio in mind, it's easier to set the image's width and the aspect ratio, and then using this to calculate for its height. This way, we can scale up or down the image by changing the image width, and it won't throw off our desired aspect ratio. We do have to make sure that when we solve for the image height the resulting height is at least 1.
> 

> [!Definition]
> **Viewport and pixel spacing**
> we also need to set up a virtual _viewport_ through which to pass our scene rays. The viewport is a virtual rectangle in the 3D world that contains the grid of image pixel locations. If pixels are spaced the same distance horizontally as they are vertically, the viewport that bounds them will have the same aspect ratio as the rendered image. The distance between two adjacent pixels is called the **pixel spacing**, and **square pixels is the standard.**

Now we need also to define the **camera center** that is a point in 3D space which all scene rays will originate (this is also commonly referred to as the _eye point_). The vector from the camera center to the viewport center will be **orthogonal** to the viewport.

![[Pasted image 20261020192338.png|321]]

Now there is a tricky part, While our 3D space has the conventions above, this conflicts with our image coordinates, where we want to have the zeroth pixel in the top-left and work our way down to the last pixel at the bottom right. This means that our image coordinate Y-axis is inverted: Y increases going down the image.

![[Pasted image 20261020192556.png|344]]

Other important things are:
- `focal_length`: this represent the distance between the origin and the viewport. 
- `viewpot_height = 2.0`: this is a trick to set the FOV to 90 degree, because:
$$\tan\left(\frac{\theta}{2}\right) = \frac{\frac{\text{viewport\_height}}{2}}{\text{focal\_length}} = \frac{1.0}{1.0} = 1.0$$
	 in this equation $\theta = FOV$, poiché l'arcotangente di $1$ è $45^\circ$ ($\frac{\pi}{4}$ rad):
$$\frac{\theta}{2} = 45^\circ \implies \theta = 90^\circ$$
	if you try to change the value of the `viewport_height` you will change the FOV, if it is bigger there will be a zoom out (**effetto quadrangolare**) effect, otherwise a zoom in (**effetto teleobbiettivo**).
### Adding a Sphere
To add any kind of object we need to calculate the intersection between the ray with this object. In this case we use a sphere. Mathematical a sphere can be describe as:
$$
x² + y² + z² = r²
$$
You can also think of this as saying that if a given point $(x,y,z)$ is on the surface of the sphere, then $x^2 + y^2 + z^2 = r^2$. If a given point $(x,y,z)$ is _inside_ the sphere, then $x^2 + y^2 + z^2 < r^2$, and if a given point $(x,y,z)$ is _outside_ the sphere, then $x^2 + y^2 + z^2 > r^2$.

If we want to allow the sphere center to be at an arbitrary point $(C_x, C_y, C_z)$, then the equation becomes a lot less nice:
$$(C_x - x)^2 + (C_y - y)^2 + (C_z - z)^2 = r^2$$
In graphics, you almost always want your formulas to be in terms of vectors so that all the $x$/$y$/$z$ stuff can be simply represented using a `vec3` class. You might note that the vector from point $\mathbf{P} = (x,y,z)$ to center $\mathbf{C} = (C_x, C_y, C_z)$ is $(\mathbf{C} - \mathbf{P})$.

If we use the definition of the dot product:
$$(\mathbf{C}-\mathbf{P}) \cdot (\mathbf{C}-\mathbf{P}) = (C_x - x)^2 + (C_y - y)^2 + (C_z - z)^2$$
Then we can rewrite the equation of the sphere in vector form as:
$$(\mathbf{C}-\mathbf{P}) \cdot (\mathbf{C}-\mathbf{P}) = r^2$$
After many steps that you can read [here](https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/ray-sphereintersection) we obtain:
$$a = \mathbf{d} \cdot \mathbf{d}$$
$$b = -2\mathbf{d} \cdot (\mathbf{C} - \mathbf{Q})$$
$$c = (\mathbf{C} - \mathbf{Q}) \cdot (\mathbf{C} - \mathbf{Q}) - r^2$$
The only unknown is $t$, and we have a $t^2$, which means that this equation is quadratic. You can solve for a quadratic equation $ax^2 + bx + c = 0$ by using the quadratic formula:
$$\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
![[Pasted image 20261020193716.png|255]]

### Surface Normals
First, let’s get ourselves a surface normal so we can shade. This is a vector that is perpendicular to the surface at the point of intersection. There are three important observations when we go to implement it, in particular regarding to have **an arbitrary length** or **will be normalized to unit length**:
1. First, if a unit-length normal vector is _ever_ required, then you might as well do it up front once, instead of over and over again “just in case” for every location where unit-length is required
2. Second, we _do_ require unit-length normal vectors in several places
3. Third, if you require normal vectors to be unit length, then you can often efficiently generate that vector with an understanding of the specific geometry class, in its constructor, or in the `hit()` function

For example, sphere normals can be made unit length simply by dividing by the sphere radius, avoiding the square root entirely. For a sphere, the outward normal is in the direction of the hit point minus the center:

![[Pasted image 20261020200644.png|425]]

### Front Faces vs Back Faces


# References
- [[Ray Tracing in One Weekend.pdf]]