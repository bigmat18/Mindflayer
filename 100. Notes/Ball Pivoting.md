**Data time:** 14:10 - 07-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Ball Pivoting

This is a derivative of [[Alpha Shapes]]. The core idea is approximate the alpha shapes just "rolling" a ball of radius $\alpha$ on the sampling S. It has the same sampling conditions ad $\alpha$-shapes holds.

The main motivations of this algorithms are:
- Alpha Shapes computation is fairly cumbersome
- May produce non manifold surfaces.

![[Pasted image 20250507141415.png | 450]]

This is a good approach because the [[Introduction to Surface Reconstruction|modern acquisition method]] hardly acquire narrow curvatures like third cases in figure above.

### Algorithm
We have and Edge($s_i, s_j$):
- Opposite point so, center of empty ball c
- Edge can be Active or Boundary

![[Pasted image 20250507142727.png | 100]]

Intial seed triangle: Empty ball of radius p passes through the three points. The **arrow** is the active edge, the **blue point** is the point on front, the **orange point** is the internal point, **red arrow** is the boundary edge.

![[Pasted image 20250507142902.png | 100]]

Ball pivoting around active edge:
![[Pasted image 20250507153651.png | 100]]
![[Pasted image 20250507153703.png | 100]]
![[Pasted image 20250507153719.png | 100]]
![[Pasted image 20250507153737.png | 100]]
![[Pasted image 20250507153837.png | 100]]
No pivot found:
![[Pasted image 20250507153908.png | 100]]
![[Pasted image 20250507153941.png | 100]]
![[Pasted image 20250507154000.png | 100]]
![[Pasted image 20250507154016.png | 100]]
# References