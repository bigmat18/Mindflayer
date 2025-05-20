**Data time:** 15:28 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory Consistency]]

**Area**: [[Master's degree]]
# Relaxed Memory Models

Each relaxed memory model allows some specific reordering of memory accesses issued by the same PE:
- $W \leftarrow R$: a LAOD can bypass an earlier STORE in program order
- $W\leftarrow W$: a STORE can bypass an earlier STORE in program order
- $R \leftarrow R$: a LOAD can bypass an earlier LOAD in program order
- $R \leftarrow W$: a STORE can bypass an earlier LOAD in program order

With SC, no ordering is allowed:
- Before a LOAD is performed wrt any other PE, all prior LOADs/STOREs must be **globally performed**
- Before a STORE is performed wrt any other PE, all prior LOADs/STOREs must be **globally performed**

![[Pasted image 20250520154349.png | 500]]

### Relaxing Strategies
##### Relaxing $W\leftarrow R$
Is one optimization proposed by [[Pipeline Processors|in-order pipelied processors]] to hide memory latencies. A later LOAD can bypass an earlier STORE.

![[Pasted image 20250520154544.png | 400]]
##### Relaxing $R\leftarrow W$
It allows a later STORE instruction to bypass an earlier LOAD instruction.

![[Pasted image 20250520154647.png | 400]]

##### Relaxing $W\leftarrow W$
It allows different STOREs (to different addresses) to be executed and completed not in program order.

![[Pasted image 20250520154754.png | 400]]

##### Relaxing $R\leftarrow R$
It allows different LOADs  to be executed in different order than the one specified by the program.

![[Pasted image 20250520154857.png | 400]]

### Relaxed Models
##### [[Sequential Consistency (SC)]]

![[Pasted image 20250520155141.png]]
All memory instructions are issued and completed in program order. All processors see the same **global ordering** of LOADs/STOREs.

##### Total STORE Ordering (TSO)

![[Pasted image 20250520155232.png]]
An earlier STORE can be reordered after a later LOAD. It allows the use of a **store buffer**. All processors see the same **global ordering** of STORES.

##### Partial STORE Ordering (PSO)

![[Pasted image 20250520155308.png]]
A **STORE** can be reordered after a next **LOAD** to a different address. A **STORE** can be reordered after a next **STORE** to a different address. **No global ordering** of instructions.

##### Weak Ordering or Relaxed Consistency (WO, RC)

![[Pasted image 20250520155358.png]]
All possible reorderings might happen and are admitted. **No global ordering** of instructions.

### Benchmarking: SC vs TSO
Performance comparison between **Sequential Consistency (SC)** and **Total Store Ordering (TSO)** on three benchmarks.

![[Pasted image 20250520155507.png | 500]]

With TSO, processors can adopt a **Write Buffer** (STOREs are completed in order but they are asynchronous with subsequent LOADs on different addresses).

# References