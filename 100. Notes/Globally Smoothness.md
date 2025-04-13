**Data time:** 18:32 - 11-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]]

**Area**: [[Master's degree]]
# Globally Smoothness

Other important things in parametrization is the globally smoothness, that is another conseguence of [[Cuts technique in Parametrization|cuts]]. The tangent directions varyes smoothly across seams. 

![[Pasted image 20250413151139.png| 200]]

 In this cases there isn't macthing between sections. To resolve this issues we need to create a parametrization **globally smooth** 
![[Pasted image 20250413151519.png | 150]]
If parametrization is globally smooth we can use it to do [[Refinement & Subdivision. Remeshing Algorithms|remeshing]]. It is particular useful for quadrangulation, need good placement of singularities. A singularity is a point where my surface is significantly different from a regular place of quadrilateral mesh.  

![[Pasted image 20250413152002.png | 350]]

In this picture, on blue vertices there are 3 quadrilateral, this is call singularity. At the opposite, the red points have more then 3 quadrilateral around vertex. The blue cases happen when we have a positive gaussian curvature. 

With this approch we can say that in each point that are not a singularity have gaussian curvature equals to 0, because they are planar areas.

# References