**Data time:** 15:54 - 24-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Parallel and distributed systems. Paradigms and models]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# False Sharing Problem

This problem arises in cases two PEs share a common cache line in their private caches. However, the two PEs modify **different words** of that line only. No sharing exists, however the cache line is the granularity of coherency action by the CC mechanism.

![[Pasted image 20250521173419.png]]

The **Solution** is design the data structures in such a way that different fields used by distinct PEs are in separated cache lines (eg, using **padding bits**)

##### Example
```c
for (int i = 0; i < m; i++) {
	y[i] = 0.0;
	for (int j = 0; j < n; j++)
		y[i] += A[i][j] * x[j];
}
```

![[Pasted image 20250524160547.png]]

- Consider m=8, and a cache line size of 64 Byte (y fits exactly in **one cache line**)
- 4 threads, one per core, each thread computing 2 elements of y
- **False sharing**: every write to y invalidates the cache line in the other core's cache. Most of these updates to y force main memory accesses 

There are two main **solutions**:
- data padding: ensures each threads's slice lay on a separate cache line
- use temporary variables: reduces the frequency of writes

![[Pasted image 20250524160608.png]]


# References