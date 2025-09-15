**Data time:** 14:47 - 10-05-2025

**Status**: #note #youngling 

**Tags:** [[3D Geometry Modelling & Processing]] [[Smoothing]]

**Area**: [[Master's degree]]
# Laplacian Smooth

With this approach we use the **diffusion flow**, it is widely used to blur images and smooth terrain surfaces. it diffuse a signal over a domain, build a scale space describing the evolution of data through time under the blurring/smoothing process.

![[Pasted image 20250510143515.png | 250]]

This method is called **Laplacian Smooth**. Where for each vertex, compute the displacement vector towards the average of its neighbors (Laplacian Operator). Then move each vertex by a fraction of its displacementet vector (diffusion over time).

![[Pasted image 20250510143754.png | 250]]

This use the base concept of diffusion flow: the neighbors (the domain) diffuse value to my point to move it proportionally. We can describe this approch in a more formal way.

Diffusion flow is modelled by the **diffusion equation**:
$$
\frac{\partial f(x,t)}{\partial t} = \lambda \Delta f(x,t)
$$
this equation is a second-order linear partial differential equation which statess that the function $f$ changes over time by a scalar diffusion coefficient $\lambda$ times its spatial [[Gradiant, Divergence and Laplacian#Laplacian|Laplacian]] 

To use this equation to smooth a manifold surface S, simply by replacing the regular Laplacian operator with the [[Discrete Differential Operators#Uniform Laplacian|Laplace-Beltrami]]. We also replace $f$ with its sample values 
$$
(f(v_1, t), f(v_2, t), \dots, f(v_n,t))^T
$$
This yields an equation:
$$
\frac{\partial}{\partial}f(v_i, t) = \lambda\Delta f(v_i, t), \:\:\:\: i = 1, \dots, n
$$
For a temporal discretizationm we divide the time axis into regular intervals of size $h$, yielding time steps $\{ t, t+h, t+2h, \dots \}$. Approximating the time derivative by finite differences:
$$
\frac{\partial f(f)}{\partial t} \approx \frac{f(t+h) - f(t)}{h}
$$
and solving for $f(t + h)$ yields the **explicit Euler integration**:
$$
f(t + h) = f(t) + h \frac{\partial f(t)}{\partial t} = f(t) + h \lambda Lf(t)
$$
with this we have now a sparse linear system has to be solved for the function values $f(t + h)$.
### Implicit Explicit
The smoothing operations can see like the resolution of linear system to minimize Laplacian
- **Explicit Euler Integration**: resolve the system by iterative substitution for small time step h
$$f(t + h) = f(t) + h\lambda Lf(t)$$
- **Implicit Euler Integration**: resolve the following linear system
$$(I - h\lambda L)f(t + h) = f(t)$$
	the system is very large but sparse.

![[Pasted image 20250510154900.png | 500]]

### Umbrella Operator
For the mesh is a little bit complicated apply Laplacian Smooth. A easy approch can be to use a simple average of neighbor, a more complex is use Umbrella Operator. For each vertex of the mesh:
$$P_{new} = P_{old} + \lambda U(P_{old})$$
where U is the Umbrella operator (the [[Discrete Differential Operators#Uniform Laplacian|Uniform Laplacian]])
$$U(P) = \frac{1}{\sum_i w_i}\sum_i w_i Q_i - P$$
![[Pasted image 20250510144308.png | 170]] ![[Pasted image 20250510144332.png | 170]]

Code from vcg/complex/algorithms/smooth.h:
```c++
static void VertexCoordLaplacianBlend(MeshType &m, int step, float alpha, bool
SmoothSelected=false)
{
	VertexIterator vi;
	LaplacianInfo lpz(CoordType(0,0,0),0);
	assert (alpha<= 1.0);
	SimpleTempData<typename MeshType::VertContainer,LaplacianInfo > TD(m.vert);
	for(int i=0;i<step;++i) {
		TD.Init(lpz);
		AccumulateLaplacianInfo(m,TD);
		for(vi=m.vert.begin();vi!=m.vert.end();++vi)
		if(!(*vi).IsD() && TD[*vi].cnt>0 ) {
			if(!SmoothSelected || (*vi).IsS()) {
				CoordType Delta = TD[*vi].sum/TD[*vi].cnt - (*vi).P();
				(*vi).P() = (*vi).P() + Delta*alpha;
			}
		}
	}
}
```

This approach allow in implement in a very simple way the smoothing by diffusion flow system, for each iteration a move a vertex in a position calculate by his neighbor, for each iteration increase smoothing and remove detail.
![[Pasted image 20250510145442.png | 400]]

### Problems
###### Shrinking
The first problems is the **shrinking**. If I start from a model and I move each vertex towards the mean of its neighbor when I have a positive [[Surfaces Curvatures|curvature]] it go to bottom because each vertex is moved to the opposite of the curvature.

![[Pasted image 20250510145807.png | 500]]

A way to minimize this issues is called [[Toubin Smoothing]]

###### Depence on Tasselation
The speed of convergence depends on mashing, where the mesh has denser tassellations needs more iterations to converge and it's slower. In the imave below the lest part converge faster.

![[Pasted image 20250510151559.png | 250]]

To fix that the use **Substitute Laplacian Operator**. Une another operator that weight vertices by considering involved edges.
$$U(P) = \frac{2}{E} \sum_i \frac{Q_i}{|e_{ij}|} - P \:\:\:\:\:\:\:\:\:\:\:\:\:where\:E = \sum_i e_{ij}$$
with this method we limit the speed of the algorithms to the most dense part.

### Mean Curvature Flow
We can use the [[Discrete Differential Operators#Cotangent formula|cotangent weight]] in the diffusion flow process. In this case we weight differently considering the different mean [[Surfaces Curvatures|curvature]].  The Laplace beltrami-operator is:
$$Hn(P) = \frac{1}{4A}\sum_i (\cot\alpha_i + \cot\beta_i)(Q_i - P)$$
What happens is the vertex will be projected along the perpendicular of the plane built on the neighbor vertex.

![[Pasted image 20250510154057.png | 150]]
This means that the initial triangulation equality will be maintain.

![[Pasted image 20250510154152.png | 500]]

# References
- [Laplacian Smoothing and Delaunay Triangulation](https://www.ljll.fr/frey/papers/meshing/Field%20D.A.,%20Laplacian%20smoothing%20and%20Delaunay%20triangulations,%201988.pdf)