**Data time:** 00:45 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Parallel and distributed systems. Paradigms and models]] [[Shared Memory Systems]]

**Area**: [[Master's degree]]
# SMP Symmetric Multi-Processor

The base memory [[Communication Latency]] is independent of the specific PE and memory macro-module. Also called **UMA (Uniform Memory Access**) Macro-modules are mutually interleaved (i.e., logically they act as a unique memory) It does not matter where you put data.

![[Pasted image 20250518004655.png]]

To distinct SMP from [[NUMA - Non Uniform Memory Access]] we can say Single-CMP machine with an off-chip P-M network connecting all the MMs with the CPU at the same distance. The distance between each PE inside the CMP and a specific MM can be different based on the distance between PEs and MINFs (and so, based on the on-chip network). However, since the distance changes only within the chip, the difference can be negligible. **The machine behaves like an SMP**.

![[Pasted image 20250518005047.png]]
### Multi-CMP Architectures
SMPs are often considered as «building blocks» to design larger shared-memory systems with very high parallelism. Multi-CMP [[NUMA - Non Uniform Memory Access]] of SMPs architectures

![[Pasted image 20250518004839.png]]
# References