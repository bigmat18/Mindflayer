**Data time:** 16:20 - 21-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Classifying Parallel Architectures]] [[High Performance Computing]]

**Area**: [[Master's degree]]
# Distributed Memory Architectures
**Processing nodes** are **complete computers** (possibly multi-processors based on multi-core CPUs potentially equipped with hardware accelerators). Each node has its own memory space non-accessible by the other nodes. Cooperation between nodes is possible only via I/O.

![[Pasted image 20250514235214.png]]

**Distributed Memory (DM)** systems are inherently [[NUMA - Non Uniform Memory Access|NUMA]]:
- each processor has its private memory (local memory), it can be an SPM or a NUMA multiprocessor.
- The address space of distinct nodes is disjoint. However, ccNUMA and COME-like multicomputers were also built in the past as both commercial products and research prototypes (example is SGI Origin 2000 series and MIT Alewife).
- Processors communicate via explicit messages through the network. We are nterested in sytems with **high-performance network topologies** and **homogeneous nodes**. Some examples are Mesh, Fat Tree, Dragonfly.
- Examples: Clusters, supercomputers, edge and cloud nodes

![[Pasted image 20250321162308.png]]
###### Key focus
Primarily on the interconnection network topology
###### Goals
**Reduce communication costs** (reducing latency, increasing available bandwidth). Communications among nodes is form of I/O for the single node.
###### Challenges
Fast messaging protocols/libraries, message routing and flow control

With this machines for example we can run a **Stencil computation**:
- Distributed memory partitioning of N x N matrix onto N processes running on M nodes, for the implementation of a 5-point stencil code. Access to the neighboring cells requires the sending and receinving of data between pairs of processes using explicit messages.

![[Pasted image 20250321170910.png | 200]]

### Programming DM systems
###### Key concept 
The primary way to exploit parallelism is through press-level parallesim, which involves running multiple process simultaneously on different nodes.
###### Reference model
**Passing Programming model** (ess POSIX socket, MPI). There were many attempts to provide a transparent shared-memory abstraction atop distributed-memory systems (the so-called SW-DSM - Software-based Distributed Shared Memory). An example of modern SW-DSM implementation is **CUDA Unified Memory** within a single node, provide a unified memory space across CPU cores and GPU cores with automatic management.
###### Challenges
- Hide communication overheads
- Explicit synchronization via messages

### Distributed Nodes 
Distributed-memory systems have **nodes** each with a **private physical addressing space**, complete I/O sub-system (with hardware accelerators), and a complete memory hierarchy. The communication point between nodes is the **I/O interface**, which is implemented by a **[[Introduction to link layer|Network Interface Card (NIC)]]**

![[Pasted image 20250514235805.png]]

The current trend assigns ever more runtime support tasks (e.g., message copy via **DMA**) to the NIC, which is often a complex **co-processor** (**smartNICs** like in **Infiniband**)
# References