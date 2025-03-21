**Data time:** 15:58 - 21-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Classifying Parallel Architectures]]

**Area**: 
# Shared Memory Architectures

The **Shared-Memory (SHM)** systems can be classified as:
##### Uniform (SMP - Symmetric Multiprocessor)
- All processors/core have equal access time to memory, same distance to shared memory.
- This organisation is also called **Uniform Memory Access (UMA)**
- Example: Single socket Intel Xeon-based system, where all cores share one memory controller
##### Non-Uniform ([[NUMA - Non Uniform Memory Access]])
- Each processor/core (or group of cores) has its local memory (usually is caches).
- Non-uniform memory access time, the memory access time is asymmetric depending on which memory is accessed (DRAM memory). Shared memory required much time.
- In **NUMA multiprocessors**, the memory is still shared among PEs, but it's physically distributed.
- Example: AMD EPYC chips, which have a NUMA organisation even in a single-socket configuration (chip composed of mutliple chiplets)

![[Pasted image 20250321160538.png]]
###### Note
We can implement a shared memory at software level but is logical shared memory
###### Key focus
Primarily on the memory organisation (memory hierarchy, processor-memory interconnections)
###### Goals
Minimise memory contention and mitigate the **Von Neumann bottleneck**
###### Challenges
Cache-coherence, memory consistency, thread synchronisation

### Programming SHM systems
###### Key concept
The primary way to program SHM systems is by exploiting the physical shared memory through thread-level parallelism
###### Reference model
**Shared Variables programming model** (Pthread, c++ threads, OpenMP). Distinct processes on the same node may communicate via shared memory buffers.
###### Challenges
- Synchronization issues
- Memory hierarchy exploitations
# References