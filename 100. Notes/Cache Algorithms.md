**Data time:** 18:48 - 23-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Cache Algorithms

It (almost) all general-purpose architectures the cache hierarchy is not explicitly managed by the user. It is managed by a set of caching policies (**cache algorithms**) that determinate which data is cached during program execution, where the data is stored, and what cache line should be evicted if the cache is full. 

Additionally, the **[[Cache Coherence Problem|cache coherence]] algorithm** keeps data coherent if there are multiple caches. There is 3 important questions that we need to answer.

### Which data should we load from main memory?
Fisrs define what is a **Cache line**: several items of information (memory words) are stored as a single memory location to enforce **[[Cache Memory|spatial locality]]**. 

Rather than requesting a single value, an entire cache line is loaded with values neighboring addresses. For **Example** with a cache line size of 64B, double precision values:
```c
//maximum of an array
//(elements stored contiguously)
for (i = 0; i<n; i++)
	maximum = max(a[i], maximum);
```
- First iteration: `a[0]` is requested resulting in a cache miss
- Eight consecutive values `a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]` loaded into the same cache line (the block factor is 8, b=8)
- The next iterations will then result in cache hits
- Subsequent request of `a[8]` results again in a cache miss, ans so on
- Overall, the hit ratio in our example is as high as 87.5% (7/8)

### Where should we store it in the cache?
Cache organized into a number of **cache lines**. Cache mapping strategy decides in which locations in the cache a copy of a particular entry of main memory will be stored.
- **Direct-Mapped Cache**: Each memory block is restricted to exactly one cache line (high miss rates, **trashing problem**, no temporal locality for cache line replacement algorithm)
- **N-Way set associate Cache**: Each memory block can be placed in any of N lines within a set, balancing hardware complexity with reduced conflict misses.
- **Fully associative Cache**: Any memory block can occupy any cache line, offering maximum flexibility but at the cost of higher complexity.

![[Pasted image 20250523191712.png]]

### If the cache is full, what data should we evict?
**Cache replacement algorithms** are used efficiently manage the limited cache space. When the cache is full, the algorithm decides which cache line to evict to make room for the new cache line that contains the requested data (Algorithms are LRU, LFU, FIFO, Random, Pseudo-LRU)
- **Least Recently Used (LRU)**: It is an eviction policy used to select one cache line to remove **according to the temporal localoty principle**. It evicts the least recently accessed cache line
	- **Pros**: Optimize workloads with temporal localoty
	- **Cons**: high overhead due to the access tracking
- **Pseudo-LRU (PLRU)**: An approximation to the LRU algorithm that reduces hardware overhead and complexity. It uses a small set of bits to keep track of which ways within a set have been recently accessed,



# References