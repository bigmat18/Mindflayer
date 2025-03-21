**Data time:** 11:23 - 12-10-2024

**Status**: #padawan #note 

**Tags:** [[3D Geometry Modelling & Processing]] [[Linear Algebra]]

**Area**: 

# Signed distances field

In math the **signed distance function** is the orthogonal distance of a given point x to a boundary of a set $\Omega$ is a metric space.
###### Definition
Let $\Omega$ be a subset of a metric space $X$ with metric $d$ and $\partial\Omega$ be its boundary. The distance between a point $x \in X$ and the subset $\partial \Omega$ is define as usual us:
$$d(x, \partial \Omega) = inf_{y \in \partial \Omega} d(x, y)$$
Where $inf$ denotes the infimum. The **signed distance function** from a point x of X to $\Omega$ is defined by
$$
f(x) = \begin{cases}
d(x, \partial \Omega ) & x \in \Omega\\
-d(x, \partial \Omega) & x \not\in \Omega
\end{cases}
$$


![[Screenshot 2024-10-12 at 11.32.23.png | 150]]

The graph (bottom, in red) of the signed distance between the points on the xy plane (in blue) and a fixed disk (also represented on top, in grey)
# References
https://en.wikipedia.org/wiki/Signed_distance_function