---
Data: 2026-10-07T16:27:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computer Vision]]"
Area: "[[Master's Degree.base]]"
---
# Light and Surfaces
We can have many lights sources that flood out world with light rays, these rays reflect surfaces, generating a field of light rays heading in all directions through space.

When a ray hit a surfaces and reflect it generally change in some attributes such **brightness** or **color**. We will describe how light interacts with surfaces and how those interactions are recorded with a **camera**

### Light Interacting with Surfaces
Visible light is electromagnetic radiation, exhibiting wave effects like diffraction. For many imaging models, however, it is helpful to introduce the abstraction of a **light ray**, describing the light radiation heading in a particular direction from a particular location in space.

![[Pasted image 20261007161240.png|356]]

The light ray is specified by: **position**, **direction**, **intensity** and **wavelength**.

The power of the outgoing light, reflected in the direction, , is determined by what is called the **bidirectional reflection distribution function** (BRDF), $F$ , of the surface
$$
\ell_{\texttt{out}} = F \left( \ell_{\texttt{in}}, \mathbf{n}, \lambda, \mathbf{p}, \mathbf{q} \right)
\tag{5.1}
$$
- $p$ direction of the light ray
- $l_{in}(\lambda)$ power of the ray
- $\lambda()$ spectral wavelength
- $q$ the direction where the light reflect
- $n$ the surface normal
#### Lambertian Surfaces
In general, BRDFs can be quite complicated, describing both diffuse and specular components of reflection. Even surfaces with completely diffuse reflections can give complicated reflectance distributions. 

A useful approximation describing diffuse reflection is the **Lambertian model**, with a particularly simple BRDF, which we denote as $F_L$. The outgoing ray intensity, $l_{out}$ , is a function only of the surface orientation relative to the incoming and outgoing ray directions, the wavelength, a scalar surface reflectance, and the incoming light power:

$$
\ell_{\texttt{out}} = F_{L} \left( \ell_{\texttt{in}} (\lambda), \mathbf{n}, \mathbf{p} \right)  = a \ell_{\texttt{in}}(\lambda) \left( \mathbf{n} \cdot \mathbf{p} \right),
\tag{5.2}
$$
- $a$ is the surface **reflectance** or **albedo**
- $n$ is the surface normal vector
- $p$ points toward the source of the incident light

> [!NOTE]
> Note that the brightness of the outgoing light ray depends on the orientation of the surface relative to the incident ray, as well as the reflectance, $a$ , of the surface. For a Lambertian surface, the intensity of the reflected light is a function of the direction of the incoming light ray, but not a function of the outgoing direction of the ray, $q$.

#### Specular Surfaces
A widely used model of surfaces with a specular component of reflection is the **Phong reflection model**. The light reflected from a surface is assumed to have three components that result in the observed reflection:
1. an **ambient component**, which is a constant term
2. a **diffuse component**, which is the Lambertian reflection
3. a **specular reflection** component

So the **phong specular contribution** for a given ray direction $q$ is:
$$
\ell_{\mbox{Phong spec}} = k_s (\mathbf{r} \cdot \mathbf{q})^\alpha \ell_{\texttt{in}},
$$
- $k_s$ is a constant
- $\alpha$ is parameter governing the spread of the specular reflection
- $r$ is a unit vector that denotes the direction of maximum specular reflection given by:
$$
\mathbf{r} = 2(\mathbf{p} \cdot \mathbf{n}) \mathbf{n} - \mathbf{p}
$$

![[Pasted image 20261007165038.png|251]]

- 𝑘𝑎: ambient light coefficient
- 𝑎 comes from Lambertian diffuse term 

**specular term:**
- 𝑘𝑠: strength of the specular reflection
- 𝛼: “shininess”, controls how tightly the highlight is concentrated
- 𝒓: direction of perfect mirror reflection of 𝒑 about 𝒏
- (𝒓 ⋅ 𝒒)𝛼 is large only when the viewing direction 𝒒 is close to 𝒓

![[Pasted image 20261007163537.png|493]]

Increasing 𝛼 **shrinks and sharpens the highligh**t (polished metal vs. matte plastic). Increasing 𝑘𝑠 **brightens the highlight** without changing its size 

![[Pasted image 20261007165237.png|373]]

the BRDF determines the image’s appearance. Geometry, lighting, and material affect the
way light interacts with surfaces, a recurring ambiguity in vision
- **advantage**: material properties can be inferred from images, and used for recognition and understanding
- **challenge**: material properties can be confused with geometry and lighting, making inference difficul
# References
- https://visionbook.mit.edu/imaging.html