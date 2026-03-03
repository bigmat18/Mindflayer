**Data time:** 14:59 - 12-08-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]]

**Area**: [[Master's degree]]
# Quadratics Error
In order to select a contraction we need some notation of the cost of contraction, to do that we define for each vertex $v = [v_x, v_y, v_z, 1]^T$ the quadratic form is:
$$
\Delta(v) = v^T Q v
$$
Now we need to compute Q in a a proper way.
### Deriving Error Quadrics
Let $n^Tv + d = 0$ be the equation representing a plane. It is based on estimating the sum of squared distances of p from all the supporting plane sof triangles in the patches. We associate a plane for each triangle involved with vertex.
$$
\Delta(v) = \Delta([v_x, v_y v_z 1]^T) = \sum_{p \in planes(v)} (p^Tv)²
$$
where $p = [a, b, c, d]^T$  represents the plane defined by the quation $ax + by + cz + d = 0$. The error metric can be rewritten as a quadratic form:
$$
\Delta(v) = v^T \bigg( \sum_{p\in planes(v)} K_p \bigg) v
$$
where $K_p$ is the matrix:
$$
K_p = pp^T = \begin{bmatrix} a² & ab & ac & ad \\ ab & b² & bc & bd \\ ac & bc & c² & cd \\ ad & bd & cd & d² \end{bmatrix}
$$
foundamental error quadratic $K_p$ can be used to find the squared distance of any point in space to the plane p.

It can also see in the following way: The squared distance of a point $x$ from the plane is
$$D(x) = x(nn^T)x + 2dn^Tx + d^2$$
This distance can be represented as a quadratic:
$$Q = (A,b,c) = (nn^T, dn, d^2) \:\:\:\:\:\:\: Q(x) = xAx + 2b^Tx + x$$
also the sum of the distance of a point from a set of planes is still a quadratic.
### Approximating Error with Quadrics
After calculating error quadrics for each vertex, for a given edge $(v_1, v_2)$ we can compute the $Q = Q_1 + Q_2$. In order to perform the contraction $(v_1, v_2) \to \bar{v}$  we must also choose a position for $\bar{v}$. A simple scheme is calculate the average point however it would be better to calculate a new value that minimize the $\Delta(\bar{v})$

Since the error function $\Delta$ is quadrati, finding the minimum is a linear problem. We find $\bar{v}$ by solving $\partial \Delta/\partial x = \partial \Delta /\partial y = \partial \Delta/\partial z$. This is equivalent to solving:
$$
\left[ \begin{array}{cccc} q_{11} & q_{12} & q_{13} & q_{14} \\ q_{12} & q_{22} & q_{23} & q_{24} \\ q_{13} & q_{23} & q_{33} & q_{34} \\ 0 & 0 & 0 & 1 \end{array} \right] \bar{v} = \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\ 1 \end{array} \right]
$$
Assuming that this matrix is invertible, we get that:
$$
\bar{v} = \left[ \begin{array}{cccc} q_{11} & q_{12} & q_{13} & q_{14} \\ q_{12} & q_{22} & q_{23} & q_{24} \\ q_{13} & q_{23} & q_{33} & q_{34} \\ 0 & 0 & 0 & 1 \end{array} \right]^{-1}  \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\ 1 \end{array} \right]
$$
if the matrix is not invertible, we attempt to find the optimal vertex along the segment $v_1v_2$.

![[Pasted image 20250407153451.png | 600]]
### QEM Algorithm
1. Compute the Q matrices for all initial vertices
2. Select all valid pairs
3. Compute the optimal contraction target $\bar{v}$ for each valid pair $(v_1, v_2)$. The error $\bar{v}^T(Q_1 + Q_2)\bar{v}$ of this target vertex becomes the cost of contracting that pair
4. Place alla pairs in a heap keyed on cost with the minium const pair at the top
5. Iteratively remove the pair $(v_1, v_2)$ of least cont from the heap, constract this pair, and update the costs of all valis pairs involveing $v_1$

# References
- [Surface Simplification Using Quadric Error Metrics](https://www.cs.cmu.edu/~garland/Papers/quadrics.pdf)