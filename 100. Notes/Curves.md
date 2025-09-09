---
Data: 2025-09-08T14:52:00
tags:
  - note
  - youngling
Connection:
  - "[[3D Geometry Modelling & Processing]]"
  - "[[Differential Geometry]]"
Area: "[[Master's degree]]"
---
# Curves

The curvature of C at P is then defined to be the reciprocal of the radius od osculating circle at point P.

![[Pasted image 20250429012827.png | 250]]

The **osculating circle** of a curve C at given point P is the circle that has the same **tangent** as C at point P as well as the same **curvature**. Just as the tangent line is the line best approximating a curved at a point P, the osculating circle is the best circle that approximates the curve at P.

In a formal way we can represent a curve in parametric form by a vector-valued function 
$$x: [a,b] \to \mathbb{R}^2\:\:\:\:\:\:\:\:\:\: x(u) = (x(u), y(u))^T$$
we consider $x, y$ differentiables functions of u. The **tangent vector** $x'(u)$ to the curve at a point $x(u)$ is defined as the first derivat of the coordinate function.

The normal value can be computed in the following way:
$$
n(u) = x'(u)^{\bot} / ||x'(u)^{\bot}||
$$

### Arc Length
The length of any curve segment defined on an interval $[c,d] \subseteq [a,b]$ can be computed as the integral of tangent vector in the following way:
$$
l(c,d) = \int^d_c ||x'(t)||dt
$$
where the tanget vector $x'$ encode the **metric** of the curve and $||x'(t)||$ is the **scalar speed** along the curve (in other word how many "meters" you do in 1 unit of $t$ parameter).

Parametric curves allow for a unique parametrization that can be defined as a length-preserving mapping, ie, an **isometry** between the parameter interval and the curve using thre reparametrization:
$$
s = s(u) = \int_a^u ||x'(t)|| dt
$$
this arc length parametrization is indipendent of the specific representation fo the curve. It maps the parameter interval $[a,b]$ to $[0, L]$ where
$$
L = l(a,b) = \int^a_b ||x'(t)||dt
$$
is the total length of the curve.
### Curvature
The curvature at point $x(s)$ can be defined:
$$
k(s) := ||x''(s)||
$$
for and arbiitrary regular curve with parametrization $u$ we can define curvature using the reparametrization acordin to arc length $s(u)$. 

Intuitively, curvature measures how strongly a curve deviates from a straight line, in other words curvatures relates the derivative of the tangent vector of curve and the curve normal vector and can ve defined using the relation
$$
x'' (s) = k(s)n(s)
$$
# References