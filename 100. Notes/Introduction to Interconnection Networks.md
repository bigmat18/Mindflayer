**Data time:** 01:52 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Introduction to Interconnection Networks

Interconnection networks are an essential component of parallel computing architectures. Different kinds of networks exist with specific properties and roles. In general, **bus-based interconnects** are not suitable for highly parallel machine.

![[Pasted image 20250518020602.png | 400]]

This solutions is chip but not effective, we have to replace it we some more structured to improve performance. In the image below we see all bus interconnection replaced by a new structure, explained following.

![[Pasted image 20250518020700.png | 400]]

From a topological perspective, interconnection networks can be represented as graphs of **nodes** and **edges** (links that convey a fixed number of bits). Nodes can be:
- **Endpoints**: terminal nodes that can be sources or destinations of paths representing valid communications.
- **Switches**: intermediate nodes responsible for forwarding messages from input links to output links. The number of input/output links of a switch is called the **switch degree**.
- **Links**: It is a physical connection (a wire) used to transfer data between endpoints,  endpoints, and switches, and between switches. High-speed serial connections (fiber optics, copper, …)

Switches implement firmware **[[Routing]] algorithms** depending on the kind of network. We will assume that the routing delay(i.e., service time and latency) paid by switches is of **one clock cycle** $\tau_{net}$. Each **link** has a fixed width (e.g., 32, 48 bits), it is unidirectional and can be occupied by one message at a time. Below a partial taxonomy of routing strategies:

![[Pasted image 20250518030256.png | 500]]

**Non-blocking networks:** in the presence of any set of currently established connections between pairs of source/destination peers, it is always possible to establish a new connection between any arbitrary unused pair of source/destination.
### Parameters in a Network

Some parameters used to evaluate a networks:
- **Path length:** number of units crossed by the considered path
- **Distance**: length of the minimal path between U1 and U2 ($d^{U1\to U2}$)
- **Average distance**: average length of all paths between two endpoints U1 and U2 (denoted by $d_{avg}^{U1\to U2}$)
- **Network average distance:** average length of the minimal paths between all pairs of communicating endpoints in the network (denoted by $d_{avg}$)
- **Network diameter:** longest minimal path over all (source, destination) pairs
- **Network Degree**: The degree (deg) of a network is the maximum number of neighbors of any n
- [[Maximum Link Load (MLL)]]
- [[Bisection Width]]
![[Pasted image 20250518023746.png | 450]]

The most important criteria to Evaluate a Network Topologies (not the only) are:
- **Low Diameter**: In order to support efficient communication between any pair of processors 
- **High Bisection Width**: A low bisection width can slow down many collective communication  operations (operations that involve a set of nodes)  and thus can severely limit the performance of  applications. However, achieving high bisection width may require a non-constant network degree 
- **Constant degree** (i.e. independent of network size): allows a network to scale to a large number of nodes without the need to add an excessive number of  connections 


### On-chip Networks
Within a CMP, several units need complec interconnections provided by **on-chip interconnection structures**, for example:
- **processor units** ([[Pipeline Processors]], [[Super-Scalar Processors]], [[Multicore Technologies]])
- **Private** and **shared caches**
- **[[Local IO]] units**
- Other kinds of on-chip units

**Proximity** between networks should be based on topologie suitable for **easy integration on-chip** (ess. wire lengths need to be short and fixed, and topologies need to be laid out in 2D on the die). 

Wires should not «consume space», i.e., they should not render the die area below unusable, i.e., low **wire density**. **Pin count**, **number of links**, and **length of links** are typical parameters when designing feasible on-chip networks.


### System-area Networks
System-area networks are used within a single parallel machine, or to connect more parallel machines to build a logically unique **[[Distributed Memory Architectures]]**.

Off-chip networks interconnecting:
- **[[Multicore Technologies]]**
- **Modular memory**
- **[[Global IO]] units** (even full co-processors like GPUs/FPGAs)
- **Complete computes** such as in clusters or multi-computers

Interconnection networks for parallel machines share several concepts with LAN/WAN but with very different trade-offs due to different time scales and requirements. **Proximity** in the order of centimeters, fractions of meters and up to tens of meters.

**Example**: a few hundred meters such as Infiniband (e.g., 120 Gbps over a distance of 300 mt). Hundreds or thousands of connected devices


### Direct and Indirect Networks
We focus on two types of networks
- **[[Direct Networks]]:** all the switches in the network are connected to possible final endpoints (blue square boxes in the figure below on the left)
- **[[Indirect Networks]]:** some switches are connected to possible final endpoints, while others act as intermediate switches in the communication paths. The difference depends on the network topology
- **Other Network Topologie**: Low-diameter topologie such as Dragonfly (diameter = 4), Slimfly (diameter = 3), Megafly and HyperX. One of this for example, the Dragonfly topology is:
	- A set of 𝑎 ≥ 2 switches forms a **group** (called **router**)
	- The switches within a group are fully connected
	- Each group (router) is connected with p processing nodes
	- Each group (router) has h links
	
	![[Pasted image 20250518223446.png | 300]]

**Example**: dragonfly formed by 8-port routers. Each router has 4 switches for a total of 9 groups (72 nodes).

![[Pasted image 20250518024750.png | 500]]

![[Pasted image 20250527175303.png]]
### Network solutions
**Buses**, **fully-connected networks**, and **crossbars** represent opposite solutions adopted in computing architectures.
##### [[Buses]]
##### [[Fully Connected]]
##### [[Crossbars]]
##### Evaluation Metrics
- **Costs of Links**
	- bus O(1)
	- crossbar O(N²): absolute maximum
- **Maximum Bandwidth**
	- bus O(1)
	- crossbar O(N): absolute maximum
- **Design complexity** to achieve the maximum bandwidth ([[MP IO Non-Deterministic|non-deterministc]] vs parallelism)
	- bus O(1)
	- crossbar $O(c^N)$: absolute maximum (monolithic design) 
- **Latency ($\infty$ distance)**
	- bus O(N)
	- crossbar O(1): absolute minimum
- **Limited-degree networks**: cost $O(N)$ or $O(N\log N)$, maximum bandwidth O(N) design complexity O(1), latency $O(\sqrt{N})$ or $O(\log N)$
 
### Communications Patters

![[Pasted image 20250527183523.png | 500]]

### Foster’s Parallel Algorithm Design Method
Conceptual framework for reasoning about parallelization of a given problem. Ian Foster proposed the **PCAM** approach:
- **Partitioning**: decompose the problem into a large amount of small (fine-grained) tasks that can be  executed in parallel.
- **Communication**: determine the required communication between tasks (dependencies)
- **Agglomeration**: combine identified tasks into larger (coarse-grained) tasks **to reduce communication by  improving data locality** (balance overhead vs. parallelism)
- **Mapping**:  assign the aggragated tasks to processes according to the network topology to minimize  ecommunication, enable concurrency, and balance **workload** 

![[Pasted image 20250527183819.png]]

###### Example: Jacobi Iteration
**[[Stencil]] code** applied on a 2-dimensional array. Used to solve 2D PDE. Update each value in the matrix with the average of its four neighbors

![[Pasted image 20250527184005.png]]

The update rule is applied iteratively until convergence. Boundary values (yellow part) remain constant at each iteration (fixed boundary conditions). At the end of each iteration, swap the updated array with the original one to avoid overwriting during the next iteration

Replaces all points of a given 2D matrix by the average of  the values around it in every iteration step until  convergence: 

```c
copy(buff, data, rows, cols);
for (int k=1; k<MaxIter; k++) { 
	for (int i=1; i<rows-1; i++) 
		for (int j=1; j<cols-1; j++) 
			buff[i*cols+j] = 0.25f * ( data[(i+1)*cols+j] + data[i*cols+j-1] 
			+ data[i*cols+j+1]+data[(i-1)*cols+j] ); 
	residual = R(buff, data); // e.g., L2-norm of the difference among cells
	if (residual < THRESHOLD) break;
	swap(data, buff, rows, cols); 
}
```

- The Initial matrix
![[Pasted image 20250527184039.png | 250]]

- The matrix after 1 iteration
![[Pasted image 20250527184113.png | 250]]

- The matrix after 25 and 75 iterations 
![[Pasted image 20250527184143.png | 500]]

There are two parallel schemes for Jacobi Iteration
- **Partitioning**: The smallest task is the computation of a single element of the Jacobi matrix
- **Communication**: Within an iteration all fine-grain tasks can be computed independently. Each task needs the data of four neighbors. At the end of each iteration, there is a synchronization barrier among  all p processors, and data is exchanged.
- **Agglomeration**: Two options proposed:
	1. by row (or by column);
	2. by using a square grid.
- **Mapping**: it follows the policy used for the agglomeration to map coarse-grain tasks to processors. By row, contiguous groups of rows are assigned to the p processors; or by square grids, rectangles of square grids are assigned to the p  processors organized in a $\sqrt{p} \times \sqrt{p}$ grid. 
	
![[Pasted image 20250527185546.png]]

Problem size (grid size) $n\times n; p$ processes running on $p$ nodes. Considering the **linear model** for the cost of communications between two processes:
$$
T_{comm}(n) = t_0 + n \cdot s
$$
- **Method 1**: each process own roughly $\frac{n}{p}$ rows to colums
$$
T_{comm}(n) \approx 2 \cdot (t_0 + n \cdot s)
$$
- **Method 2**: each process owns a $\frac{n}{\sqrt{p}} \times \frac{n}{\sqrt{p}}$ sub-block
$$
T_{comm}(n) \approx 4 \cdot \bigg(  t_0 + \bigg( \frac{n}{\sqrt{p}}\bigg) \cdot s\bigg)
$$
 Method 2 superior for large p since communication time decreases with p while it remains constant for Method 1.
# References