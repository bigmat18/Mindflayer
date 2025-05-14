**Data time:** 01:36 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Introduction to Data Parallelism

**Data parallelism** is a very general parallelization paradigm. It can be applied both on [[Stream Parallelism|stream]] and a sigle inputs scenarios. It can improve both the [[Ideal Service Time]] and [[Communication Latency]] of the sequential program.

![[Pasted image 20250514013916.png | 400]]

However, the **generality** of data parallelism comes with addition challenges
- How are data structures **partitioned** or **replicated**?
- How are output data structures **collected**?
- How are workers organized with each other?
- Are workers **fully independent** (map) or do they **exchange messages** (stencils)?

One of the challenging aspects to apply data parallelism is its inherent **complexity**, which derives from the large space of **variants** of data-parallel parallelizations that often exist for the same problem.

![[Pasted image 20250514014111.png | 500]]

One of the main characteristic of data parallel is **data partitiong** and **function replication**
# References