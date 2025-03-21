**Data time:** 14:12 - 21-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Classifying Parallel Architectures]]

**Area**: [[Master's degree]]
# SIMD (Single Instruction, Multiple Data)

- All PEs/CPUs execute the same instruction at any fiven clock cycle on a differenct set of data in parallel. This model is relate to **data parallelism**, when you want to compute something on multiple data elements.
- Each CPU operates on different data streams, and usually each CPU has an associated data memory module.
- The execution is **synchronous** in **locksteps** 
- Some examples are:
	- Vector units of modern **pipeline/supersclasar CPUs**
	- GPUs: they implement SIMD execution within Streaming Multiprocessors, However, at a broader architectural level, the follow the **SIMT model**, which generalises SIMD by supporting thread divergence and independent execution paths.

![[Pasted image 20250321144651.png]]
# References