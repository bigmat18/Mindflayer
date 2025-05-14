**Data time:** 01:49 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Collective communications]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Scatter
The **problem** si the follow: given a data structure $A_i$ a module S sends distinct partitions of A to a specified set of destination modules $D_0, \dots, D_{n-1}$

**Example**. A process S has an array of size L elements. Each destination $D_i$ receives the i-th partition. A first solution is base on a **sequential implementation** of the scatter.

![[Pasted image 20250514135127.png | 550]]

### Tree-Structured Scatter
Analogously to che case of the [[Multicast]] we can design a **k-array tree-based version** of the **scatter** collective. The first solution (**binary tree**) exploits dedicated modules (ie processes)

![[Pasted image 20250514135348.png | 500]]

- The reduction of the **[[Ideal Service Time]]** is modest, however the **[[Communication Latency|latency]]** becomes logarithmic.
- A higher **arity** of the tree increases latency and ideal service time (however, we need fewer processes)

### Farm-Based Scatter
Analogously to the [[Multicast]] collective, we can design a **[[farm]]-like parallelization** of the scatter applied on stream. Each worker is in charge of scattering each received input (e.g., an array) to all the destination modules. The number of workers should be found to eliminate the [[Bottleneck Parallelization|bottleneck]] (i.e., its [[Optimal Parallelism Degree]])

![[Pasted image 20250514140614.png | 600]]
# References