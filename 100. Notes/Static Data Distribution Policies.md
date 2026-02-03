---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# Static Data Distribution Policies

Data parallelism is a program model where the same operations (function) is applied in parallel to multiple independent **partitions** of a data set. The parallelism is exploited through a set of Workers, each computing the elements of the partition. Distinct partitions of the dataset can be computed independently by Workers ([[Map Parallelization]]) or, instead their computations requires interactions (i.e., message exchanges or synchronizations) among Workers ([[Stencil]]).

**Data partitioning** is the strategy of dividing input data among available threads or processes.
- Data partitioning directly influences the efficiency and performance of data parallel computations

A key challenge arises when Worker’s workload is not evenly distributed (**workload imbalance**)
- Some threads/processes will finish later, causing other threads/processors to wait idle
- Effective workload balancing techniques are crucial to ensure balanced workload distribution, optimal resource use, and overall performance improvement

###### Example: Mandelbrot Set
The Mandelbrot set is the set of all complex numbers $c\in\mathbb{C}$ for which the sequence defined by the iteration:
$$
Z(c) = \begin{cases}Z_0=0 \\ Z_{n+1}= Z²_n + c\end{cases}
$$
remains bounded as $n \to \infty$. In practice, it is checked whether $|z_n(c)|\leq 2$ for some large number of iterations (e.g., 1000). If this holds, it is assumed that $c$ is part of the Mandelbrot set.

To obtain a nice fractal, the color of each pixel depends on the number of  iterations required until  $|z_n(n)|>2$. The color of each pixel of the figure can be computed independently (embarrassingly data parallel computation)

![[Pasted image 20260202151419.png | 250]]

Black pixels (i.e., points belonging to the Mandelbrot set) require more iterations than others (colored) pixels, thus, the computational load is unbalance

 What would happen if we computed the Mandelbrot set using 3 threads, partitioning the pixels as shown in the figure?
 
![[Pasted image 20260202151452.png | 550]]


### Data Distribution Policies
Key question: How can we aggregate data into tasks and assign them to the available processing elements (Workers) so as to ensure balanced computation?

In general, the assignment of tasks can be done **statically** or **dynamically**:
- **In the static task assignment**
	- Input data collection is partitioned once at the beginning, before the computation starts 
	- Each Worker receives a fixed partition and processes it independently
	- The final output is partitioned accordingly (output partitions might be gathered or reduced into an output collection)
	- Typically, each partition has roughly the same size (or better yet, a similar computational cost)
	
- **In the dynamic task assignment**
	- Input data collection is divided in small partitions (chunks) 
	- Chunks are assigned to Workers on-demand (or they fetch tasks dynamically based on their current workload)

###### Static Policies
- The focus is to identify a data partitioning that evenly distribute the workload
- The ideal use case has a regular workload
- No dynamic task scheduling overhead
- Standard policies: block, cyclic, block-cyclic
- Example: Dense Matrix-Vector product, Regular Stencil-based Computation, Convolution, etc.

###### Dynamic Policies
- Adapt tasks assignment to Workers to handle irregular workloads
- Improved efficiency for workloads with skewed computational demands
- Additional overhead due to tasks scheduling (i.e., extra messages/synchronizations for coordination)
- Standard policicies: on-demand, work-stealing
- Example: Mandelbrot set, graph-based algorithms, N-Body Simulations, Sparse Matrix-Vector product, … 

##### Matrix-Vector Multiplication Example
$$
A \in \mathbb{R}^{m\times n} \text{ and } x \in \mathbb{R}^n \:\:\: b_i = \sum_{j=0}^{n-1} A_{ij} \cdot x_j \text{ for all } i\in {0, \dots, m-1}
$$
Simple analysis using CREW PRAM:
- **First try**: 
	- We can use $p = m\times n$ processors for computing each single $A_{ij}$ in $O(1)$ then in $O(\log n)$  executing the $m$ reductions
	- the cost is $C(p) = O(m\times n \times \log n)$, thus exploiting the parallelism at the level of the single task is not cost-optimal
- **Second Try**:
	- We can use $p=m$ processors each computing in $O(n)$ a local dot-product
	- The cost is $C(p) = O(m\times n)$ which is cost optimal.

 So the minimal task granularity is the computation of one $b_i$. We need $A_{i*}$ (i.e., i-th row of A) and $x_*$  (i.e., all $x$). Each $b_i$ (i.e., each task) can be computed independently. In real world scenarios, m and n are large numbers (i.e., >> num cores of the machine), therefore to maximize single processor utilization, each processor computes a partition of k tasks

![[Pasted image 20260202155147.png]]

- **Block task distribution**: the block size is at least $\lfloor \frac{m}{p} \rfloor$. If ($m \mod p$) = $k \neq 0$ then $\forall p_i | i < k$ the block size is $\lceil \frac{m}{p} \rceil$ 
![[Pasted image 20260202155507.png]]
![[Pasted image 20260202155341.png]]

- **Cyclic task distribution**: the task $t_i$ is assigned to $p_{i \mod p}$

![[Pasted image 20260202155525.png]]
![[Pasted image 20260202155544.png]]

- **Block-Cyclic task distribution**: Given a block of size $c > 0$ ($p \cdot c$ is called **stride**) then $t_i$ is assogmed to processor $p_{(i \div c) \mod p}$ . The cylic ditribution has $c=1$
![[Pasted image 20260202160205.png]]
![[Pasted image 20260202160216.png]]

Test on the front-end node of the spmcluster: 10 repetitions for each point, min and max  removed and then computed the average. Variance no reported in the plot. 

$m = 2^{18}, n = 2^{15}$ sequential version takes about $9.54s$. The block-cyclic distribution (bc) uses $c=8$ as the chunk size. Why is the efficiency low?
- $E(20) = 8.22/20 \approx 41\%$ 
- $OI = \frac{2 \cdot m \cdot n}{8 (mn + n + m)} \approx 0.25$ FLOPS/byte
- low operational intensity; the algorithm is memory bound

![[Pasted image 20260202162128.png | 500]]


# References