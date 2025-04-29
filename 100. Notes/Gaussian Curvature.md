**Data time:** 01:53 - 29-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Differential Geometry]]

**Area**: [[Master's degree]]
# Gaussian Curvature

Defined as $K = k_1 \cdot k_2$
- $> 0$ when the surface is a sphere. This is a point goes outwards or inwards. Curvatures with same sign.
- $0$ if locally flat (also called **developable**)
- $< 0$ for hyperboloids. This happen when in this point the curvatures have opposite sign.

![[Pasted image 20250429015710.png | 270]]

A point x on the surface is called:
- **Elliptic**: if $k > 0$ ($k_1$ and $k_2$ have the same sign)
![[Pasted image 20250429021233.png | 150]]

- **Hyperbolic**: if $k < 0$ ($k_1$ and $k_2$ have opposite sign)
![[Pasted image 20250429021302.png | 150]]

- **Parabolic**: if $k = 0$ (exactly one of $k_1$ and $k_2$ is zero)
![[Pasted image 20250429021321.png | 150]]

- **Planar**: if $k = 0$ (equivalently $k_1 = k_2 = 0$)
![[Pasted image 20250429021340.png | 150]]

In a **torus** example:
![[Pasted image 20250429021414.png | 400]]

### Developed surfaces
An interesting property is a surface is a **developed surface** iff $K = 0$. Flatting introduce no distortion.

![[Pasted image 20250429021703.png | 300]]

### Intrinsic / extrinsic
Gaussian curvature is an **intrinsic** properties of the surface (even if we defined in an extrinsic way). It is possible to determinate it by moving on the surface  keeping the geodesic distance constant to radius r and measuring the circumference $C(r)$
$$K = \lim_{r\to 0} \frac{6\pi r - 3C(r)}{\pi r³}$$

# References