**Data time:** 23:35 - 17-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory and Local IO]]

**Area**: [[Master's degree]]
# Memory Atomicity

An additional feature be provided for the memory in multi-processors is a **individual sequence of memory accesses**. A first solution (assuming no caches) is an additional bit (**indivisibility bit**, shortly called INDIV) is associated with each request. If it is **1**, once the request is accepted by the memory, the other requests coming from other PEs are left pending until **INDIV** is reset to **0** by the same PE.

![[Pasted image 20250517224248.png]]

To do that there are two solution: or a **special annotations** in LOADs/STOREs or **special atomic RAW instructions**. The RAW instructions are quite complex to be interpreted by the machine.

Their interpretation involves caches and coherency protocols. Here we assume that memory only is involved in this process.

![[Pasted image 20250517233339.png | 500]]
- We can assume each MM is equipped with proper **units buffering** requests coming from other PEs during indivisible sequences (the reality is different)
- Atomicity is provided with the granularity of the cached line size.
# References