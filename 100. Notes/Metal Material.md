---
Data: 2026-10-24T01:31:00
Tags:
  - note
  - youngling
Connection:
  - "[[Ray Tracing Walktrough]]"
Area:
  - "[[Books.base]]"
---
# Metal Material

For polished metals the ray won’t be randomly scattered. The key question is: How does a ray get reflected from a metal mirror? Vector math is our friend here:

![[Pasted image 20261024143522.png|299]]

- The reflected ray direction in red is just $v+2b$
- $n$ is a unit vector (length one), but $v$ may not be

To get the vector $b$, we scale the normal vector by the length of the projection of $v$ onto $n$, which is given by the dot product $v⋅n$ (we assume that $n$ is a unit vector, otherwise we need to normilize it).

Finally, because v points _into_ the surface, and we want b to point _out_ of the surface, we need to negate this projection length. Putting everything together, we get the following computation of the reflected vector:

```c++
inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}
```

### Fuzzy Reflection
We can also randomize the reflected direction by using a small sphere and choosing a new endpoint for the ray. We'll use a random point from the surface of a sphere centered on the original endpoint, scaled by the fuzz factor.

![[Pasted image 20261024143834.png|214]]

The bigger the fuzz sphere, the fuzzier the reflections will be. **This suggests adding a fuzziness parameter that is just the radius of the sphere (so zero is no perturbation)**. The catch is that for big spheres or grazing rays, we may scatter below the surface. We can just have the surface absorb those.

```c++
bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
const override 
{
    vec3 reflected = reflect(r_in.direction(), rec.normal);        
    reflected = unit_vector(reflected) + (fuzz * random_unit_vector());        
    scattered = ray(rec.p, reflected);
    attenuation = albedo;        
    return (dot(scattered.direction(), rec.normal) > 0);    
}
```


# References