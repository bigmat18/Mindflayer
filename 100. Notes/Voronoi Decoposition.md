**Data time:** 16:39 - 28-04-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Sampling]]

**Area**: [[Master's degree]]
# Voronoi Decoposition

**Voronoi** is a recurring pattern in Nature structures. The main idea of this decoposition is a discrete set of entities competing for resources.

In a formal way we consider a set P of points on the plane $P = \{p_0, \dots, p_i, \dots, p_n\}$. The **Voronoi Diagram** is a partition of the plane. Consider for each point $p_i$ the region of the plane closest to $p_i$.

![[Pasted image 20250428164426.png | 200]]

In the classic definition there are bounded and unbounded. Regions are convex polygons bounded by half plane intersections.

![[Pasted image 20250428164607.png | 200]]

The [[Remeshing Introduction|dual]] of the Voronoi Diagram is a **nice** triangulation (called Deloni triangulation) of the point set. We connect each points adding an edge. Each triangle has the **empty-circle** property, if we do a circle that crosses the three points it not contains any other point, this means the triangulation is well done.

![[Pasted image 20250428165146.png | 300]]

The Voronoi Diagram is a partition of the plane, and for this it can be defined for every plane. **Nice** voronoi diagrams came from **nice** point distributions. How to get nice point sets?

### Centroidal Voronoi Diagrams
This is a type of VD generated from random seeds are not well places around the seeds. Region are not "centered" around the seeds.

![[Pasted image 20250428170633.png | 200]]

We need to move the seeds towards the centroid of the region.

![[Pasted image 20250428170739.png | 200]]
After we use the new barycenter to calculate a new voronoi diagram.

![[Pasted image 20250428170839.png | 200]]
this aproch must be iterated n times to convergenve a nice voronoi diagram.
#### Lioyd's method
The Lioyd's method is based on:
1. Generate the Voronoi tessellation $V(s)$ in $\Omega$
2. move each site $s_i \in S$ to the centroid $p_i$ of the corresponding Voronoi region $V_i \in V$
3. if the new sites in S meet some convergence criterion, then termiante; otherwise return to step 1.

#### Centroid Voronoi Tasselation (CVT)
Methods that iterativeky optimize the position of the samoples using some energy function. 

A CVT is a VT where each site (point) lies in the centroid of its region:
$$p_i = \frac{\int_{V_i}x\rho(x) dx}{\int_{V_i}\rho(x)dx}$$
With $\rho(x)$ some [[Probabilità sulla retta reale|density function]]. The miminum of the energy function below is on CVT:
$$F(S,V) = \sum_{i=1}^n \int_{V_i} \rho(x) |x - s_i|² dx$$
![[Pasted image 20250428174232.png]]

There is methiods that modify CVT to obtein better sampling properties. **Capacity Constrained CVT** for example is CVT plus the contraint that each region of the VT has the same area.

![[Pasted image 20250428174409.png]]

### Voronoi Diagrams over surfaces
We just nee a simple way to define the closest concept. We usually use [[PDS on Surface|geodesic]] that is the shortest path on the surface connecting two points.

![[Pasted image 20250428174643.png | 200]]

More difficult is relaxing with Loyd's method. Because we need calculate the centroyd of a pace of surface, this is not a easy task because **centroid is not well defined**, it could be outside the surface.

- Switch to the energy minimizing center: new site is the point having minimal sum of squared distances from all the points of the region.
- Another way is constraining seeds on the boundary during relaxation. This fit very well with geodesic distance

![[Pasted image 20250428175133.png |400]]


Its easy biasing the Voronoi Diagram, the **dentisy** of the pattern can be controlled (like for [[Poisson Disk Sampling (PDS)]] we could be control the ray), weight the distance according to a scalar field.

![[Pasted image 20250428175451.png | 400]]

Beyond density we can control the **orientation**, We can bias the shape of the cells: we use a frame field to define deformation. The general solution is deform the space of the domain until the defined metric is isotropic (change uniformly in each directions).

![[Pasted image 20250428175651.png | 150]]
# References