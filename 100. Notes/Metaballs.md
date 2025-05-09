**Data time:** 02:54 - 09-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Surface Recostruction]]

**Area**: [[Master's degree]]
# Metaballs

It's a technique that allow to go from a point cloud to surface rapresentation. We assume that the point in input not only are surface point but also the inside of object. The core idea is each point generate a ball and for each point we have a $f$ that is the sum of function that have maximum in the points and decay with distance.
$$f(x_i) = 1 \:\:\:\:f(R) = 0  \:\:\:\:f'(x_i) = 0 \:\:\:\:f'(R)= 0$$
$$f(x) = \sum_i\bigg(2\frac{r³}{R³} - 3\frac{r²}{R²} + 1\bigg) \:\:\:\:r=||x - x_i|| \:\:\:\:R=support\:radius$$

![[Pasted image 20250509030312.png|350]]

This methods has the problem that many time dont create a surface that go through the points but a surface near the points. The solution was the [[Radial Basis Functions (RBF)]]
# References