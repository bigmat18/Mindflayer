**Data time:** 12:18 - 03-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Refinement & Subdivision. Remeshing Algorithms]]

**Area**: [[Master's degree]]
# Butterfly Algorithm

This is an algorithms that use **triangular meshes**, it is also **primal** and **interpolating** (we mantein the original position of original vertex). 

![[Pasted image 20250403122006.png | 250]]

We add vertex on edge using the following pattern:
$$E_1 = \frac{1}{2}(d_1 + d_2) + \omega (d_3 + d_4) - \frac{\omega}{2}(d_5 + d_6 + d_7 + d_7) \:\:\:\:\:\: d'_i = d_i$$

![[Pasted image 20250403170944.png | 350]]

This type of algorithms suffers from **ringing problems**. This problems does that if we have a straight surface it can became wavy, like the image above.  Instead in [[Loop Schema Algorithms]] this property is maintained.
# References