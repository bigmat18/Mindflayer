**Data time:** 18:52 - 27-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Sampling]]

**Area**: [[Master's degree]]
# Samples generation

Another approach very easy to implement ii based to change the order of things to do. In the surface that we want sampling first we insert n points random (n >> m where m is the actual number of points that we want), and then we start to remove a point, for each point that we have remove we also delete the points too close to it (with the distance < r). This can be efficiently with a [[Uniform Grid]]

In different worlds:
1. Generate the pool: a dense sampling of the surface.
2. Until there are available samples:
	1. Pick a sample randomly
	2. Remove the sample closer than he disk radius from the pool

We can choose a sample to remove with a **heuristics** like: remove the sample that remove the lower number of other points. In this way I privilege the choice to the next points.

A **Question** is "Both versions work well: where is the trick?" The answer is the creation of the pool essentially converts the initial "continuous" domain (faces) in a point sampled one. The algorithm can not guarantee maximality of the sampling up to the inter-sample distance of the initial pool.
# References