**Data time:** 14:27 - 23-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Cache Memory

In Modern CPUs typically contain a hierarchy of two or three levels of cache (L1, L2, L3). Caches have higher bandwidth and lower latency compared to main memory but much smaller capacity.

Trade-off between capacity and speed:
- Example: the L1 cache is small but fast (**0.5-1 ns**), while the L3 cache is relatively large but much slower than L1 (**15-40ns**)
- Main memory access time is higher (**50-100 ns**)

![[Pasted image 20250523143201.png]]

Caches could be private for a single core or shared between several cores. Following image show a cache organization used in today's CMPs.

![[Pasted image 20250523143321.png]]

###### Example Architecture
Simplified model to establish an **upper bound on performance** (will never go faster than what the model predicts). Let's suppose:
- 1 CPU with 8 cores @3GHz capable of 16 **floating point operations per core per clock cycles (FLOP)**. The peak performance is:
$$
Rpeak = 3 \cdot 8 \cdot 16 = 384 GFLOP/s
$$
- The DRAM peak memory transfer rate is 51.2 GB/2
- **On-chip fast shared cache pf capacity 512 KB**

![[Pasted image 20250523143922.png | 350]]

###### Example of Matrix Multiplication of above Architecture
```c
//Matrix Multiplication (nxn)
for (int i = 0; i<n; i++)
	for (int j = 0; j < n; j++) {
		double dotp = 0;
		for (int k = 0; k<n; k++)
			dotp += U[i][k]*V[k][j];
		W[i][j] = dotp;
	}
```
We have a $W = U \times V$ square matrices with $n=128$. Total size of the matrices is $128² \cdot 3 \cdot 8B = 384KB$ **(first in cache)**. Data transfer time (from/to cache) is
$$
t_{mem} = \frac{384 KB}{ 51.2 GB/s} = 7.5 \mu s
$$
Let's assume we transfer the input matrices U and V from memory to cache once at the beginning, and the matrix W from cache to memory once at the end.

The **Computation time** is:
$$
t_{comp} = \frac{2^{22} FLOP}{384 \:GFLOPs} = 10.4 \mu s
$$
The total operations are $2 \cdot n \cdot n² = 2 \cdot 128³ = 2^{22} FLOP$. In this case $t_{mem} < t_{comp}$ thus MM is **compute bound** (for n=128). 

The **Execution time** is:
$$
t_{exec} \geq 7.5 \mu s + 10.4 \mu s = 17.9 \mu s
$$
Achievable performance (upper bound): $\frac{2^{22} FLOP}{17.9 \mu s} = 223 \:GFLOPS$ ($\approx$ 60% of peak). What if matrices are bigger that cache?

### Locality Principle
The locality principle is the **driving force** that makes the memory hierarchy (caches) work properly. It increases the probability of reusing data blocks that were previously moved from level n to level n-1 (i.e., closer to the CPU), thus reducing the miss rate.

- **Temporal Locality (data reuse)**: Temporal locality refers to the property of a program to repeatedly access the same memory locations over a short period of time. **Property of the data access pattern**; the cache mapping strategy (direct cache vs associative cache) and the [[Cache Algorithms|replacement algorithm]] (LRU, Random) have an impact on temporal locality
- **Spatial Locality**: Spatial locality refers to the property of **a program to access memory locations that are spatially close to each other**
	- **Enforced by moving data in blocks** between levels of the memory hierarchy (cache line, page)
	- The typical cache line size is **64 bytes** (i.e., it stores 8 – 16 contiguous memory words)

###### Terminology
ff the data the processor requests is present in one block at the closest memory level, it is called a **hit**. Otherwise, it is a **miss**, and the next memory level is accessed to retrieve the block containing the requested data.
- **Cache hit**: The data is present in the cache 
- **Cache miss**: The data is not present in the cache
- **Miss penalty**: the time spent transferring a cache line into the first level cache and the requested data to the processor
- **Miss rate (MR)**: the time spent transferring a cache line into the first level cache and the requested data to the processor
$$
MR = \frac{\#misses}{\#of \:memory\:accesses}
$$
- **Hit rate (HR)**: $HT = 1 - MR$

###### Measuring $CPU_{time}$ with caches
$$
CPU_{time} = ClockCycles \cdot ClockCycleTime = IC \cdot CPI \cdot ClockCycleTime
$$
- IC (Instruction Count) is the number of program instructions executed
- IC can be furhter detailed as $IC_{CPU} + IC_{MEM}$, the former are ALU instructions (eg register - register), the latter are memory access intructions (eg, load, store)
- CPI is the average ClockCycles Per Instruction, and is defined as $\frac{ClockCycles}{IC}$
$$
CPI = \bigg(\frac{IC_{CPU}}{IC} \bigg) \cdot CPI_{CPU} + \bigg(\frac{IC_{MEM}}{IC} \bigg) \cdot CPI_{MEM}
$$
	where $CPI_{CPU}$ are the average cycles per ALU instruction and $CPI_{MEM}$ are the average cycles per memory instruction

	Considering that each memory instruction may generate a cache hit or miss with a given probability, and given the HitRate the probability of a cache hit, we have:
$$
	CPI_{MEM} = CPI_{MEM-HIT} + (1 - HitRate) \cdot CPI_{MEM-MISS}
$$
###### Example of impact of data locality on CPU time
Let's consider the following scenario:

![[Pasted image 20250523154211.png]]

The **Question** is: What is the impact of the $CPU_{time}$ if the cache hit rate (HitRate) drops from 95% to 60%?
$$
CPU_{time}(sec) = IC \bigg[  \bigg( \frac{IC_{CPU}}{IC}\bigg) \cdot CPI_{CPU} + \bigg(\frac{IC_{MEM}}{IC}\bigg) \cdot (CPI_{MEM-HIT} + (1-HitRate) \cdot CPI_{MEM-MISS}) \bigg] \cdot ClockCycleTime
$$
If we replace the value in the formula we have $\frac{CPU_{time}(HitRate=0.6)}{CPU_{time}(HitRate = 0.95)} = 4.5$

A decrease in the cache hit rate from 95% to 60% (about 37%) resulted in a 4.5X increase is execution time. **It is therefore critical to keep the cache hit rate high to maximize performance**.

### Cache Working Set
The **working set (WS)** of a program, for a given a memory hierarchy, **is the collection of data the program actively accesses during a specific time interval**.

For **Example**: Consider a memory hierarchy with levels M1 (eg cache) and M2 (eg main memory) and a program P. If the entire working set of P fits in M1, the probability of cache misses is minimized.

The important points:
- A program’s working set usually changes over time as it accesses different parts of its data
- The size and composition of the working set depend on the program’s memory access patterns (e.g., sequential vs. random access). 

### Optimizing Cache Accesses: GEMM

```c
//Matrix Multiplication
for (int i = 0; i<n; i++)
	for (int j = 0; j < m; j++) {
		float accum = 0;
		for (k = 0; k < l; k++)
			accum += A[i*l+k]*B[k*n+j];
		C[i*m+j] = accum;
}
```

Matrix multiplication **textbook algorithm (naive)**: $C_{n\times m} = A_{n\times l} \times B_{l\times m}$ stored in linear arrays in **row-major order**. Access pattern:
- The matrix A is accessed contiguously $(i,k) \rightarrow (i, k+1)$. This i-th row of A is accessed m times.
- The matrix B is accessed non-contiguously $(i,k) \rightarrow (i+1, k)$. The entire matrix B is accessed n times. Elements of B are "far apart" in main memory ($l \times sizeof(float)$ distant), **thus they are not stored in the same cache line for large l**. Cache lines containing B elements can be evicted from L1/L2-cache before they are reused in subsequent iteration of j for large l.

![[Pasted image 20250524142115.png]]

The Working set of the textbook GEMM algorithm is:
- One row of A, one column of B and one element of C. This is the minimum working set that minimizes cache misses for a relatively small time interval.
- However, if we consider a large enough time interval (ie considering the for j loop), the working set is **one row of A, one row of C and the entire matrix B**. For large l and m, B will not fit into LLC, thus what we can do is to improve cache locality by exploiting at least spatial locality for B.

```c
//Transpose-and-Multiply
for (int k=0; k<l; k++)
	for (int j = 0; j<m; j++)
		Bt[i*l+k] = B[k*n+j];
for (int i=0 ; i<n; i++)
	for (int j=0; j<m; j++) {
		float accum = 0;
		for (int k=0; k<l; k++)
			accum += A[i*l+k] * Bt[j*l+k];
		C[i*m + j] = accum;
}
```

**Transpose-and-multiply**: $Bt_{m\times l} = (B_{l\times m})^T$ and $C_{n \times m} = A_{n \times l} \times Bt_{l \times m}$
- Access pattern of A contiguously: (i, k) -> (i, k+1)
- Accesses pattern of Bt contiguously: (j, k) -> (j, k+1)

This new algorithm leverages spatial locality for A, and B (i.e., Bt). It introduces an extra overhead for transposing B.

![[Pasted image 20250524143759.png]]

To reduce the working set to fit into the cache we can **change the algorithm** and use a **block-based** (more often called **tile-based**) GEMM:
- Instead of working on the entire matrix at once, we can split it into smaller sub-matrices (tiles) of TxT
- The Working set consists of a tile from each (A, B, C). These tiles exhibit spatial and temporal locality
- If the tree tiles fit into the cache we minimize the cache misses.

T should be chosen such that 3 blocks fit in the lower-level cache. Typical values range from 16 to 128, but you should benchmark and profile your code!
$$
T \approx \Bigg\lceil \sqrt{\frac{Cachesize}{3 \times sizoof(data)}} \Bigg\rceil
$$
```c
//Tile-based Matrix Multiplication (MM)
for (int i = 0; i < m; i += T)
	for (int j = 0; j < n; j += T)
		for (int k = 0; k < l; k += T)
			// Compute MM for the current tile
			for (int ii = i; ii < std::min(i + T,m); ++ii)
				for (int jj = j; jj < std::min(j + T,n); ++jj) {
					float sum = 0.0;
					for (int kk = k; kk < std::min(k + T,l); ++kk)
						sum += A[ii * l + kk] * B[kk * n + jj];
					C[ii * n + jj] += sum;
}
```

![[Pasted image 20250524144628.png]]

![[Pasted image 20250524144700.png]]


# References