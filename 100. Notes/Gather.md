**Data time:** 01:49 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Collective communications]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Gather

The **problem** is given a set of data structures $A_0, A_1, \dots, A_{n-1}$ local to a set of source modules $S_0, S_1, \dots, S_{n-1}$, a module G builds a unique data structure A merging the partitions $A_0, A_1, \dots, A_{n-1}$.

**Example**: $A_0, A_1, \dots, A_{n-1}$ are partitions of an array
$$L_{gather} = T_{id-gather} = T_{send}(L)$$
![[Pasted image 20250514132039.png | 450]]

- The cost of the **send** of each $S_i$ to $G$ is included in the **[[Ideal Service Time]]** of $S_i$
- The **[[Communication Latency|latency]]** of the gather considers one sends only (all are in parallel)
- Gather implementation can be done without additional copies if the [[Basics of Message Passing|message-passing]] runtine is proprerly implemented.
- In the code $A + i$ denotes the logical address of the $(i+1)$-th element of A.

### All-Gather Collective
The previous gather collective has been described in its basic form (very common in data-prarallel programs). Other variants of it exists and can be used in specific data parallel patterns. One of them is the **All-Gather Collective**:

![[Pasted image 20250514132746.png | 150]]

- Each module of the left produces a **partition** of a data structure
- Each partition is **[[Multicast]]** to all modules on the right
- All modules on the right **assemble** the partitions to build the same whole data structures.

The [[Ideal Service Time]] of the sources includes the cost of the **[[Multicast|multicast collective]]** ($m>0$ send primitives of size $L/n$). The ideal service time of the destinations includes the impact of the gather overhead (small if we can **avoid extra copies**).
# References