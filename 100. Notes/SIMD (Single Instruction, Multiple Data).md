**Data time:** 14:12 - 21-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Classifying Parallel Architectures]]

**Area**: [[Master's degree]]
# SIMD (Single Instruction, Multiple Data)

SIMD is a computing paradigm related to [[Introduction to Data Parallelism|Data parallelism]]. In SIMD machines the target is:
- fine-grained arithmetic, fixed or floating-point operations
- [[Map Parallelization|Map]] paradigm, notably for loops on large arrays
- All PEs/CPUs execute the same instruction at any fiven clock cycle on a differenct set of data in parallel. This model is relate to **data parallelism**, when you want to compute something on multiple data elements.
- Each CPU operates on different data streams, and usually each CPU has an associated data memory module.
- The execution is **synchronous** in **locksteps** 
- Some examples are:
	- Vector units of modern **pipeline/supersclasar CPUs**
	- GPUs: they implement SIMD execution within Streaming Multiprocessors, However, at a broader architectural level, the follow the **SIMT model**, which generalises SIMD by supporting thread divergence and independent execution paths.

![[Pasted image 20250321144651.png]]

In principle, any data parallel program can be applied by a SIMD machine although **maps** are very common. The SIMD conecpts has been applied in computing architectures in different ways.
- **Vectorization facilities of modern Processing Elements**: (i.e., CPU cores): vectorized EUs in [[Pipeline Processors]]/[[Super-Scalar Processors]] with ISA extensions like SSE (Intel), AVX (Intel), Neon (ARM), and others.
- **[[Array Processors]]**: standalone implementation of the SIMD paradigm at the firmware level applied by specialized architectures with high parallelism.
- **[[Graphical Processing Units (GPU)]]**: IMD is applied more broadly together with **MIMD + hardware multithreading**. This combination is often referred with the commercial name of Single Instruction Multiple Threads (SIMT)
- **[[SIMT|Multiple Threads (SIMT)]]**
# References