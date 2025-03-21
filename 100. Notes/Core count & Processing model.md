**Data time:** 16:26 - 21-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Classifying Parallel Architectures]]

**Area**: [[Master's degree]]
# Core count & Processing model

Another type of classification is based on cores count. Considering the number of cores (thus FLOPS) in general-purpose MIMD machines.
- $O(10^1 \div 10^2)$ cores, for a single multiprocessor chip (CMP).
	The 4th generation of AMD EPYC CPU has 64 cores (128 threads)
- $O(10^2 \div 10^3)$ cores, for a shared-memory tightly-coupled multiprocessor
	HPE Superdome Flex series, large ccNUMA multiprocessor.
- $O(10^3 \div 10^5)$ for Distributed-Memory loosely-coupled systems, from small to large compute clusters.
	Small-medium cluster with 16-128 nodes up to large cluster where nodes have one or more GPUs
- $O(10^5 \div 10^6)$ top supercomputers
	Example the Leonardo supercomputer

# References