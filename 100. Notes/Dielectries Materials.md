---
Data: 2026-10-24T14:41:00
Tags:
  - note
  - youngling
Connection:
  - "[[Ray Tracing Walktrough]]"
Area:
  - "[[Books.base]]"
---
# Dielectries Materials

Clear materials such as water, glass, and diamond are dielectrics. When a light ray hits them, **it splits into a reflected ray and a refracted (transmitted) ray**. We’ll handle that by randomly choosing between reflection and refraction, only generating one scattered ray per interaction. A quick review of the terms:
- **Reflected**: is ray hits a surface and then **bounces** off is a direction
- **Refracted**: is a ray bends as it transitions from a material's surroundings into the material itself (as with glass or water). This is why a pencil looks bent when partially inserted in water.

The amount that a refracted ray bends is determined by the material's _refractive index_. Generally, this is a single value that describes how much light bends when entering a material from a vacuum.

### Snell's Law
The refraction is described by Snell’s law:
$$\eta \cdot \sin\theta = \eta' \cdot \sin\theta'$$
Where:
- $\theta$ and $\theta'$ are the angles from the normal 
- and $\eta$ and $\eta'$ (pronounced “eta” and “eta prime”) are the **refractive indices**. 

The geometry is:
![[Pasted image 20261027122219.png|316]]

In order to determine the direction of the refracted ray, we have to solve for $\sin\theta'$:
$$\sin\theta' = \frac{\eta}{\eta'} \cdot \sin\theta$$
On the refracted side of the surface **there is a refracted ray $R'$ and a normal $n'$**, and there **exists an angle, $\theta'$**, between them. We can split **$R'$ into the parts of the ray that are perpendicular to $n'$ and parallel to $n'$:**
$$R' = R'_{\perp} + R'_{\parallel}$$
If we solve for $R'_{\perp}$ and $R'_{\parallel}$ we get:
$$R'_{\perp} = \frac{\eta}{\eta'} (R + \vert{}R\vert{}\cos(\theta)n)$$
$$R'_{\parallel} = -\sqrt{1 - \vert{}R'_{\perp}\vert{}^2} \, n$$
You can go ahead and prove this for yourself if you want, but we will treat it as fact and move on. We know the value of every term on the right-hand side except for $\cos\theta$. It is well known that the dot product of two vectors can be explained in terms of the cosine of the angle between them:
$$a \cdot b = \vert{}a\vert{}\vert{}b\vert{}\cos\theta$$
If we restrict $a$ and $b$ to be unit vectors:
$$a \cdot b = \cos\theta$$
We can now rewrite $R'_{\perp}$ in terms of known quantities:
$$R'_{\perp} = \frac{\eta}{\eta'} (R + (-R \cdot n)n)$$
When we combine them back together, we can write a function to calculate R′:
```c++
inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = std::fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}
```

### Total Internal Reflection
One troublesome practical issue with refraction is that **there are ray angles for which no solution is possible using Snell's law.** When a ray enters a medium of lower index of refraction at a sufficiently glancing angle, it can refract with an angle greater than 90°.


> [!error] Problem
> If we refer back to Snell's law and the derivation of $\sin\theta'$:
$$\sin\theta' = \frac{\eta}{\eta'} \cdot \sin\theta$$
If the ray is inside glass and outside is air ($\eta = 1.5$ and $\eta' = 1.0$):
$$\sin\theta' = \frac{1.5}{1.0} \cdot \sin\theta$$
The value of $\sin\theta'$ cannot be greater than $1$. So, if,
$$\frac{1.5}{1.0} \cdot \sin\theta > 1.0$$
the equality between the two sides of the equation is broken, and a solution cannot exist. If a solution does not exist, the glass cannot refract.

Here all the light is reflected, and because in practice that is usually inside solid objects, it is called _total internal reflection_. This is why sometimes the water-to-air boundary acts as a perfect mirror when you are submerged — if you're under water looking up, you can see things above the water, but when you are close to the surface and looking sideways, the water surface looks like a mirror.

We can solve for `sin_theta` using the trigonometric identities:
$$\sin\theta = \sqrt{1 - \cos^2\theta}$$
and
$$\cos\theta = R \cdot n$$
So the code will became:
```c++
double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

if (ri * sin_theta > 1.0) {
    // Must Reflect
    ...
} else {
    // Can Refract
    ...
}
```

### Schlick Approximation
Now real glass has reflectivity that varies with angle, look at a window at a steep angle and it becomes a mirror. There is a big ugly equation for that, but almost everybody uses a cheap and surprisingly accurate polynomial approximation by Christophe Schlick. This yields our full glass material:

```c++
static double reflectance(double cosine, double refraction_index) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0*r0;
    return r0 + (1-r0)*std::pow((1 - cosine),5);
}
```

# References