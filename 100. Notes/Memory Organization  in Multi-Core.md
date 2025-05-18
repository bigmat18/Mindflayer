**Data time:** 15:53 - 21-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Classifying Parallel Architectures]] [[High Performance Computing]]

**Area**: [[Master's degree]]
# Memory Organization in Multi-Core

This model classify system on the memory system. Here we implicitly refer to [[MIMD (Multiple Instruction, Multiple Data) |MIMD]] parallel architectures. Considering the **memory system**, we have:

![[Pasted image 20250321155712.png]]
### [[Shared Memory Architectures]] (called multiprocessors)

### [[Distributed Memory Architectures]] (called multi-computers)

### SHM vs DM systems
Distributed memory systems are more scalable, more costly, and less energy efficient. A modern high-end CMP node has more computing power than a supercomputer of 10 years ago at a fraction of the cost and power consumed.

From the standpoint of the **runtime system programmer**
- For **Shared Memory Systems**, the physical shared memory can be used directly for fast synchronization and communication between processes/threads. However, efficient management of locking and syncronization is generally a critical point.
- For **Distributed Memory Systems** the most importat aspect is to reduce the cost of communications much as possible, and it is possible with **overlapping of I/O and computation**, reducing memory copies for I/O, using fast messaging protocols (ess RDMA - Remote Direct Memory Access).

### Message Passing (MP) vs Shared-Variable (SV) programming models
###### Generality
MP is more general because it can be used in both DM and SHM systems. For example MPI-based parallel code can be executed efficiently on both DM and SHM systems. While in SHM systems MP can be implemented using SV.
###### Scalability
MP is generally mode scalable than SV.
###### Intrusiveness and Complexity
- MP is more intrusive as it requires explicit communication for data movement and synchronization
- More verbose and error-prone code
- In the SV model, managing synchorinzation to avoid race condition can still be complex and error-prone.
###### Performance
- For MP depend on the communication layer
- For SV depends on the memory access contention and in general to the memory hierarchy.
# References
