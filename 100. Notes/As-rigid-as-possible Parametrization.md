**Data time:** 17:47 - 13-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Parametrization]] [[Parametrization Techniques]]

**Area**: [[Master's degree]]
# As-rigid-as-possible Parametrization

This technique work with a **local-global approach**, it do steps of optimization in a local context and steps of optimization in a global context. 
1. For each triangles create a parametrization (local) that it isn't correct. Each individual triangle is independently flattened into plane without any distortion.

![[Pasted image 20250413175941.png | 400]]

2. Merge in UV space (averaging or more sophisticated strategied) and try to minimize the error.

![[Pasted image 20250413180050.png | 400]]
3. Iterate this two steps until we found a good solution.

![[Pasted image 20250413180218.png | 500]]

In many cases this approach works very well because minime [[Parametrization Distortion|distotions of areas]].
# References