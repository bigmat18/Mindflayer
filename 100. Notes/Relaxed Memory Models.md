**Data time:** 15:28 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Atomic Operations & Memory Consistency]]

**Area**: [[Master's degree]]
# Relaxed Memory Models

**Relaxed memory consistency models** permit certain orderings to be violated. Why? To gain performance by hiding memory latency.
- Overlapping memory access operations with other independent operations

Each relaxed memory model allows some specific reordering of memory accesses issued by the same PE:
- $W \leftarrow R$: a LAOD can bypass an earlier STORE in program order
- $W\leftarrow W$: a STORE can bypass an earlier STORE in program order
- $R \leftarrow R$: a LOAD can bypass an earlier LOAD in program order
- $R \leftarrow W$: a STORE can bypass an earlier LOAD in program order

With SC, no ordering is allowed:
- Before a LOAD is performed wrt any other PE, all prior LOADs/STOREs must be **globally performed**
- Before a STORE is performed wrt any other PE, all prior LOADs/STOREs must be **globally performed**

![[Pasted image 20250520154349.png | 500]]

Every modern processor uses **Write Buffers** to improve performance:
- A processor may reorder its read ahead of its pending writes ($W_x \to R_y$) to hide write latency
- This is perfectly fine from the viewpoint of a single-thread control flow
- This is a standard memory reordering optimization used by most modern processors

![[Pasted image 20260202211714.png | 200]]

A **Write Buffer** is a small queue or memory area in the CPU that holds data from recent write operations which have **not yet been fully committed to main memory** (or the next level in the  memory hierarchy). By storing these “pending” writes in a buffer, the **processor can continue  executing subsequent instructions without having to wait for each store to complete**. If the  CPU needs to read a location that’s in the write buffer, it can **forward that data directly from the buffer** instead of waiting for it to reach cache or main memory.
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

##### Total STORE Ordering (TSO) and Processor Consistency (PC)

![[Pasted image 20250520155232.png]]
An earlier STORE can be reordered after a later LOAD. It allows the use of a **store buffer**. All processors see the same **global ordering** of STORES.

**TSO**: All writes from each processor are seen in the same order by all processors. Loads in a processor may bypass pending stores in a write buffer (load cannot bypass store to the same memory location!):
- All processors observe the same order for writes from any processor
- A load may bypass an earlier store (to a different address), potentially causing a thread to read an older value
- x86 uses the TSO memory consistency model. NOTE: TSO allows r1=0 and r2=0 in the previous example!

 **PC**: Writes to the same memory location are seen in the same order by all processors. However, writes to **different locations may be observed in different orders by different processors**
- It allows the processor to reorder writes across different memory locations, potentially leading to higher performance
- Requires explicit synchronization (e.g., memory fences) when coordinated updates across multiple locations are needed
- No modern architecture strictly implement PC. Some systems further relax memory ordering.

In general:
- Store→Store: **non** riordinati (ordine degli store preservato)
- Load→Load: **non** riordinati (in genere)
- Load→Store: **non** riordinati
- **Store→Load: può** essere riordinato (il classico caso “store buffering”
##### Partial STORE Ordering (PSO)

![[Pasted image 20250520155308.png]]
A **STORE** can be reordered after a next **LOAD** to a different address. A **STORE** can be reordered after a next **STORE** to a different address. **No global ordering** of instructions. In other word relaxed also $W_X \to W_Y$

Why? The processor might reorder write operations in the Write Buffer for performance reasons (e.g., one 
write might be a cache miss, while the other might be a cache hit whose management costs less).
- This is a valid optimization if a program consists of a single instruction stream
- For multiple threading applications additional synchronization is needed

**Why do architecture allow more aggressive reordering?**
- Overlap multiple independent reads and writes in the memory system
- Execute reads early and delay writes, further hiding memory latency
- Out-of-order execution and speculative execution allow the CPU to keep its pipelines full

in general:
- **Store→Store: può** riordinare se le store sono a indirizzi diversi
- Store→Load: può riordinare (come TSO)
- Altri riordini dipendono dalla specifica, ma il punto distintivo è S→S rilassato

##### Weak Ordering or Relaxed Consistency (WO, RC)

![[Pasted image 20250520155358.png]]
All possible reorderings might happen and are admitted. **No global ordering** of instructions. ARM and POWER micro architectures adopt a very relaxed memory model for better performance, at the cost of increased programming complexity

### Benchmarking: SC vs TSO
Performance comparison between **Sequential Consistency (SC)** and **Total Store Ordering (TSO)** on three benchmarks.

![[Pasted image 20250520155507.png | 500]]

With TSO, processors can adopt a **Write Buffer** (STOREs are completed in order but they are asynchronous with subsequent LOADs on different addresses).

# References