**Data time:** 19:29 - 23-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallelization methodology and metrics]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Cache Write Policies

When a CPU writes data to the cache, the value in the cache may be **inconsistent (the data is no coherent)** with the value in main memory.

- **Write-though policy**: caches handle this by updating the data in the main memory when it is written to the cache (always implemented with a **store/write buffer**)

- **Write-back policy**: Caches mark data in the cache as dirty (one extra bit per cache line needed). When a new cache line from memory replaces the cache line that has the dirty bit set, the "dirty" cache line values are written to memory. A **store/write buffer** is generally used to reduce the cost of cache accesses.

![[Pasted image 20250524145143.png]]
# References