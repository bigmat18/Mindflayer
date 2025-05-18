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

Switches implement firmware **[[Routing]] algorithms** depending on the kind of network. We will assume that the routing delay(i.e., service time and latency) paid by switches is of **one clock cycle** $\tau_{net}$. Each **link** has a fixed width (e.g., 32, 48 bits), it is unidirectional and can be occupied by one message at a time. Below a partial taxonomy of routing strategies:

![[Pasted image 20250518030256.png | 600]]

**Non-blocking networks:** in the presence of any set of currently established connections between pairs of source/destination peers, it is always possible to establish a new connection between any arbitrary unused pair of source/destination.

Some parameters used to evaluate a networks:
- **Path length:** number of units crossed by the considered path
- **Distance**: length of the minimal path between U1 and U2 ($d^{U1\to U2}$)
- **Average distance**: average length of all paths between two endpoints U1 and U2 (denoted by $d_{avg}^{U1\to U2}$)
- **Network average distance:** average length of the minimal paths between all pairs of communicating endpoints in the network (denoted by $d_{avg}$)
- **Network diameter:** longest minimal path over all (source, destination) pairs

![[Pasted image 20250518023746.png | 450]]
##### On-chip Networks
Within a CMP, several units need complec interconnections provided by **on-chip interconnection structures**, for example:
- **processor units** ([[Pipeline Processors]], [[Super-Scalar Processors]], [[Multicore Technologies]])
- **Private** and **shared caches**
- **[[Local IO]] units**
- Other kinds of on-chip units

**Proximity** between networks should be based on topologie suitable for **easy integration on-chip** (ess. wire lengths need to be short and fixed, and topologies need to be laid out in 2D on the die). 

Wires should not «consume space», i.e., they should not render the die area below unusable, i.e., low **wire density**. **Pin count**, **number of links**, and **length of links** are typical parameters when designing feasible on-chip networks.
##### System-area Networks
System-area networks are used within a single parallel machine, or to connect more parallel machines to build a logically unique **[[Distributed Memory Architectures]]**.

Off-chip networks interconnecting:
- **[[Multicore Technologies]]**
- **Modular memory**
- **[[Global IO]] units** (even full co-processors like GPUs/FPGAs)
- **Complete computes** such as in clusters or multi-computers

Interconnection networks for parallel machines share several concepts with LAN/WAN but with very different trade-offs due to different time scales and requirements. **Proximity** in the order of centimeters, fractions of meters and up to tens of meters.

**Example**: a few hundred meters such as Infiniband (e.g., 120 Gbps over a distance of 300 mt). Hundreds or thousands of connected devices
### [[Maximum Link Load (MLL)]]

### [[Bisection Width]]

### Direct and Indirect Networks
We focus on two types of networks
- **[[Direct Networks]]:** all the switches in the network are connected to possible final endpoints (blue square boxes in the figure below on the left)
- **[[Indirect Networks]]:** some switches are connected to possible final endpoints, while others act as intermediate switches in the communication paths. The difference depends on the network topology

![[Pasted image 20250518024750.png | 500]]

### Network solutions
**Buses**, **fully-connected networks**, and **crossbars** represent opposite solutions adopted in computing architectures.
##### Buses
![[Pasted image 20250518024855.png]]
All N units are connected through a shared broadcast medium Lowest link cost (only one link) **Maximum [[Processing Bandwidth|bandwidth]]** O(1), **[[Communication Latency]]** O(N) (due to the arbitration delay), **node degree** 1 It is a blocking network with diameter 1 and bisection width 1.
##### Fully Connected
![[Pasted image 20250518025026.png]]
Directly connects all N units with a single hop. Highest link cost $O(N²)$ and pin count issues **Maximum bandwidth** $O(N)$, **latency** $O(1)$, node degree $N-1$ It is a **non-blocking network** with **diameter** 1 and **bisection** width $N²/4$ (with N even).
##### Crossbars
![[Pasted image 20250518025038.png]]
Directly connects all N units in the blue set with all N units in the red set. Highest link cost $O(N²)$ and pin count issues **Maximum bandwidth** $O(N)$, **latency** O(1), **node degree** N It is a non-blocking network with **diameter** 1 and **bisection width** $N²/2$ (with N even) For moderate values of N, it can be implemented by a single **firmware unit** (switch) with parallelism at the clock cycle level.

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
 
 
# References