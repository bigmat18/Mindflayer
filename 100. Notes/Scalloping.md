**Data time:** 17:02 - 27-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Sampling]]

**Area**: [[Master's degree]]
# Scalloping

This is a variance of Dart Throwing that allow to have a convergence in $O(n\log{n})$. The core idea is: if a sampling is not maximal, there must be an available location in the neighborhood of the unavailable region.

![[Pasted image 20250427170619.png | 550]]
Intuitively:
1. We start from a set of samples, and we store the boundary of regions that we have covered (the boundary data structure is a queue of circle arc)
2. If I put a new point on the boundary queue we must found a way to modify the queue, and this is a simple problem

$O(n\log{n})$ because we assume to have a data structure to index the boundary with access time to $O(\log{n})$ and repeat the access for each sample, $n$ times.

Scalloped regions:
- **Scalloped sector**: a region of the domain bounded by two circular arcs
- **Scalloped region**: a union of scalloped sectors.

![[Pasted image 20250427171709.png | 450]]

### Adding a new disk
$$N_p = D(p, 4r) - \bigcup_{p'\in P}\begin{cases}D(p', 4r) & p'<p\\D(p', 2r) &p'\geq p\end{cases}$$
Where $N_p$ is available neighborhood, and D is the disk. $p'<p$ and $p'\geq p$ is used for order of insertion.

![[Pasted image 20250427172705.png | 500]]

### Speed
The maximum number of scalloped sectors of r neighboorhod is bounded by a constant. The update of neighboorhood is limited to the $4r$ radius from the sample and can be done in O(1).

**Unbiased sampling**. All available neighborhoods are stored in a balanced tree $O(\log N)$


# References