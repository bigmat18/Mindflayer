---
Data: 2026-10-23T21:44:00
Tags:
  - note
  - youngling
Connection:
  - "[[Ray Tracing Walktrough]]"
Area:
  - "[[Books.base]]"
---
# Diffuse Materials
Now that we have objects and multiple rays per pixel in [[Antialiasing in Ray Tracing]], we can make some realistic looking materials. We will see:
- A **Simple diffuse material**
- The real **Labertions corection**

### Simple Diffuse Material
Diffuse objects that don’t emit their own light merely take on the color of their surroundings, but they do modulate that with their own intrinsic color. Light that reflects off a **diffuse surface has its direction randomized**, so, if we send three rays into a crack between two diffuse surfaces **they will each have different random behavior**:

![[Pasted image 20261023231301.png|287]]

They might also be absorbed rather than reflected: The darker the surface, the more likely the ray is absorbed.

Let's start with the most intuitive: a surface that **randomly bounces a ray equally in all directions**. For this material, a ray that hits the surface has an **equal probability of bouncing in any direction away from the surface**.

![[Pasted image 20261023231454.png|303]]

we need to figure out how to manipulate a random vector so that we only get results that are on the surface of a hemisphere. There are analytical methods of doing this, but they are actually surprisingly complicated to understand, and quite a bit complicated to implement. Instead, we'll use what is typically the easiest algorithm: **A rejection method**

> [!NOTE] Definition
> A rejection method works by **repeatedly generating random samples until we produce a sample that meets the desired criteria**. In other words, keep rejecting bad samples until you find a good one.

There are many equally valid ways of generating a random vector on a hemisphere using the rejection method, but for our purposes we will go with the simplest, which is:
1. Generate a random vector inside the unit sphere (in ower case using **rejection method**)
2. Normalize this vector to extend it to the sphere surface
3. Invert the normalized vector if it falls onto the wrong hemisphere

To check if the random vector is inside the sphere we will:
1. pick a random point inside the cube enclosing the unit sphere (that is, where x, y, and z are all in the range $[−1,+1]$). If this point lies outside the unit sphere, then generate a new one until we find one that lies inside or on the unit sphere.

![[Pasted image 20261024010010.png|326]]

This is an example code:
```c++
inline vec3 random_unit_vector() {
    while (true) {
        auto p = vec3::random(-1,1);
        auto lensq = p.length_squared();
        if (lensq <= 1)
            return p / sqrt(lensq);
    }
}
```

Now that we have a random unit vector, we can determine if it is on the correct hemisphere by comparing against the surface normal:
2. We can take the **dot product of the surface normal and our random vector** to determine if it's in the correct hemisphere. 
	- If the dot product is positive, then the vector is in the correct hemisphere. 
	- If the dot product is negative, then we need to invert the vector.

```c++
inline vec3 random_on_hemisphere(const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}
```

### Labertian Reflection
Scattering reflected rays evenly about the hemisphere produces a nice soft diffuse model, but we can definitely do better. A more accurate representation of real diffuse objects is the _Lambertian_ distribution

> [!NOTE] Definition
> This distribution scatters reflected rays in a manner that is proportional to $cos(ϕ)$, where $\phi$ is the angle between the reflected ray and the surface normal.

This means that a reflected ray **is most likely to scatter in a direction near the surface normal**, and less likely to scatter in directions away from the normal. This non-uniform Lambertian distribution does a better job of modeling material reflection in the real world than our previous uniform scattering.

![[Pasted image 20261024010838.png|288]]

We can create this distribution by adding a random unit vector to the normal vector. At the point of intersection on a surface there is:
- the hit point, p
- and there is the normal of the surface, n

Moreover, this surface has exactly two sides, so there can only be two unique unit spheres tangent to any intersection point. These two unit spheres will be displaced from the surface by the length of their radius, which is exactly one for a unit sphere.
- One sphere will be displaced in the **direction of the surface's normal** (n)
- one sphere will be displaced in the opposite direction (-n)

From this:
- one of the spheres will have its center at (P+n). That is consider **inside**
- and the other sphere will have its center at (P−n). That is consider **outside**

We want to select the tangent unit sphere that is on the same side of the surface as the ray origin. Pick a random point S on this unit radius sphere and send a ray from the hit point P to the random point S (this is the vector $(S−P)$)

 We should be able to notice two important visual differences:
1. The shadows are more pronounced after the change
2. Both spheres are tinted blue from the sky after the change

Both of these changes are due to the less uniform scattering of the light rays—more rays are scattering toward the normal.
- This means that for **diffuse objects,** they will appear _darker_ because less light bounces toward the camera. 
- For the **shadows**, more light bounces straight-up, so the area underneath the sphere is darker.

### Gamma Correction
The picture is very dark, but our spheres only absorb half the energy of each bounce, so they are 50% reflectors. The spheres should look pretty bright (in real life, a light grey) but they appear to be rather dark.

![[Pasted image 20261024012442.png]]

If you look closely, or if you use a color picker, you should notice that the 50% reflectance render (the one in the middle) is far too dark to be half-way between white and black (middle-gray). Indeed, the 70% reflector is closer to middle-gray.****

> [!error] Problem
> The **reason** for this is that almost all computer programs assume that an image is “**gamma corrected**” before being written into an image file. This means that the 0 to 1 values have some transform applied before being stored as a byte.

> [!NOTE] Definition
> Images with data that are written without being transformed are said to be in _linear space_, whereas images that are transformed are said to be in _gamma space_. We need to go from linear space to gamma space, which means taking the inverse of “gamma 2", which means an exponent of 1/gamma (in own case with **gamma 2**).

![[Pasted image 20261024012931.png]]

# References