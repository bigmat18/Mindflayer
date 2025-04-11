**Data time:** 17:46 - 11-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]]

**Area**: [[Master's degree]]
# Cuts technique in Parametrization

Clearly needed for closed surfaces. For simple case, like a sphere, it's enough do a cuts long a meridian through which the sphere becomes a disk. Obviously the cuts can be more complex, usually more cute bring to less distortion.

![[Pasted image 20250411175649.png | 500]]

For a generic surface the question is how many cuts we will need?
- For a [[Representing real-world surfaces|genus]] 0 surface: any tree of cuts is allowed
![[Pasted image 20250411180526.png | 300]]

- For a genus 1 surface: two looped cuts
![[Pasted image 20250411180605.png | 300]]

- For a genus 3 surface: 6 looped cuts
![[Pasted image 20250411180635.png | 200]]

- For a genus $n$ surface: $2n$ looped cuts
![[Pasted image 20250411180713.png | 200]]

## Generic Cut Strategies
##### Unstructured cuts
We do unstructured cuts based on [[Parametrization Distortion|distortion]].

![[Pasted image 20250411181552.png | 450]]
##### Per Quad
If we have a quad mesh much cores we can do a decomposition based of quad, they are simple to pact and are easy to interpolate each other. 

![[Pasted image 20250411183050.png | 450]]
##### Implicit
There are approaches that do an implicit cuts, we build a parametrization instead of from 3D object to surface we do from 3D object to other 3D object that is a cube object.

![[Pasted image 20250411183123.png | 350]]
##### Regular Cuts 
We decompose own surface into a sets of triangles and than map them.

![[Pasted image 20250411183153.png | 350]]
# References