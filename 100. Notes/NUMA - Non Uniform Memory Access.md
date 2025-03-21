**Data time:** 16:06 - 21-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Classifying Parallel Architectures]]

**Area**: [[Master's degree]]
# NUMA - Non Uniform Memory Access

Multi-socket server architectures and also single-socket (also called **chiplets**):
- A NUMA node corresponds to an entire CPU (socket) or a a chiplet with its local memory and set of cores. It also attached to a shared memory.
- Each socket (or chiplet) has its memory controller and local memory
- To maintain a single, shared address space, the nodes are connected by a high-speed Network on Chip (NoC)
- The OS, tries to allocate memory "close to" where the allocating thread is running to reduce memory access costs.
- Each socket (or chiplet) has a shared Last Level Cache (LLC) usually L3.

###### NUMA multicore example: spmnuma

![[Pasted image 20250321161632.png]]

This is a two socket NUMA multiprocessor, if we see internally each CPU are compose by 4 nodes each one have 8 core (4 real core + 4 with hypertheading). L1 and L2 are privare caches and L3 is shared.
# References