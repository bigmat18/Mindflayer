**Data time:** 01:47 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Collective communications]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Multicast

The **problem** is if a module sends the **same** message to a specified set of destinations modules (or even all). For example a process S trasmits a message of size L to a set of $n > 0$ destination processes $D_0, \dots, D_{n-1}$

A first solution is based on the **sequential multicast implementation**:

![[Pasted image 20250514015555.png | 450]]

Multicast **[[Ideal Service Time]]** is equal to multicast **[[Communication Latency|latency]]**. The **cost model** is:
$$T_{id-multicast} = L_{multicast} = n\cdot T_{send}(L) = n \cdot (T_{setup} + L \cdot T_{transm})$$
### Tree-structured Multicast
Multicast cam impair the scalability of a parallel program. We often need to paralellize it (it might become a **[[Bottleneck Parallelization|bottleneck]]**). A solution with dedicated modules performing the mutlicast according to a **binary tree structure**.

![[Pasted image 20250514020619.png | 500]]

A **k-ary tree** instead of a binary tree can be used to reduce the number of modules ie $\#modules = n - 1 / k-1$. The [[Ideal Service Time]] becomes $T_{id-multicast} = k \cdot T_{send}(L)$, [[Communication Latency|latency]] instead becomes $L_{mutlicast} = \lceil\log_K(n)\rceil \cdot k \cdot T_{send}(L)$

### Farm-based Multicast
Multicast on different stream elements can be parallelized as a [[Farm]] (provided that we are working on streams). The number of workers should be found in such a way as to eliminate the bottleneck in the multicast stage.

![[Pasted image 20250514021106.png | 450]]

We need m+1 modules (processes) to perform the multicast in parallel (they use the same number of CPU cores). [[Communication Latency]] is exactly as with the sequential multicast (as usual for a farm, it cannot improve the latency)
### Multicast done by Workers
A tree-structured multicast can be implemented without additional **dedicated modules** (processes). The tree can be mapped onto the set of destination modules according to a distributed strategy

![[Pasted image 20250514022101.png | 600]]

The communication topology is based on a **depth-first strategy** (other strategies are possibile) where:
- Node 0 is the root
- Each node j at level $0< i \leq \log_2n$ sends to nodes j+1 and $j + 2^{i-1}$
The **[[Ideal Service Time]]** of the destination modules includes the overheads of two communication it they are not masked ie:
$$T_{id-Di} = \max\{T_{calc}, 2 \cdot T_{send}(L)\}$$

 
 
# References