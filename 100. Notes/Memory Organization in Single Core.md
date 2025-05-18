**Data time:** 22:31 - 17-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory and Local IO]]

**Area**: [[Master's degree]]
# Memory Organization in Single Core

From a logical perspective, the memory of a [[Shared Memory Architectures|shared-memory]] system is like a **centralized** entity that
- it is capable of receiving **indipendent requests** from the PEs of machine (for reading or writing a cache line)
- it tests in every clock cycle $\tau_M$ the presence of a request coming from any PE and processes them **[[MP IO Non-Deterministic|non-deterministic]]**

![[Pasted image 20250517223540.png]]

Non-determinism can be implemented as **real parallelism** in the same clock cycle
- if more input requests (coming from different PEs) are **compatible** (for different addresses, different lines) and
- if **memory [[Processing Bandwidth|bandwidth]]** is sufficient, more compatible memory requests can be served simultaneously.

### [[Memory Atomicity]]

### Memory Technologies
The main technologies to implement **Random Access Memory** (RAM) are the following:
##### [[Dynamic RAM (DRAM)]]

##### [[Static RAM (SRAM)]]

##### [[Memory Controllers (MINF)]]

### Address Mapping Schemes
A system can feature several DIMM Macro-Modules (MM) to provide a high-capacity and high-[[Processing Bandwidth|bandwidth]] memory. The question is: How can we distribute the cache lines among the different MMs?
##### Solution 1: Sequential Memory Organization

![[Pasted image 20250518001225.png | 500]]

Suppose $m>1$ MMs each having a capacity of $C>0$ cache lines of $\sigma = 64$ bytes each. The memory contains $C\cdot m$ lines. The **most $\lceil \log_2 m \rceil$ significant bits** of the physical addresses identify the MM.
##### Solution 2: Mutually-Interleaved Organization

![[Pasted image 20250518002144.png | 500]]

Suppose $m>1$ MMs each having a capacity of $C>0$ cache lines of $\sigma = 64$ bytes each. The memory contains $C\cdot m$ lines. The **last $\lceil \log_2 m \rceil$ significant bits** of the physical addresses identify the MM.

### Memory Contention
When the machine is almost idle, MINFs do not have memory requests in their **queues**. Every request is sent immediately to the corresponding **MM**. When the workload is **memory-bound** and the memory bandwidth is not sufficient, such queues grow and saturate. This implies higher [[Communication Latency]] of memory requests to be served since there is a long waiting queue of pending requests.
 
 ![[Pasted image 20250518002620.png|200]]
- This situation is further worse in case of frequent **bank conflicts**
- We experience a **bank conflict** if two consecutive accesses target the same bank within a DRAM chip in a rank
- The **pre-charge phase** is costly and increases memory utilization (higher response time)

### Asymptotic Evaluation
Let us simplify a lot of the memory sub-system behavior. Suppose that each MM is sequential and serves one request ($\sigma = 64$ bytes) at a time with a given latency/service time $L_M$. Suppose:
- $n>0$ identical PEs issuing requests to the memory (each request is delivered to one MM for 64 bytes)
- $m > 0$ identical MMs (mutually-interleaved organization)

Under this assumptions, the probability of any PE to access any MM is uniformly distributed $p = 1/m$. Let $p(k)$ be the probability that $k>0$ distinct PEs over n, are trying to simultaneously access the same MM. It is distributed with the **[[Variabili Aleatorie Notevoli|binomial law]]**
$$p(k) = \binom{n}{k}\bigg( \frac{1}{m}\bigg )^k \bigg( 1 - \frac{1}{m} \bigg)^{n-k}$$
For each MM $j =1, \dots, m$ let $Z_i$ be a binary random variable taking value 0 with probability $p(0)$, 1 with probability $1-p(0)$

The mean value of the random variable $Z_j$ is $E(Z_j) = 1 - p(0) = 1-(1 -1/m)^n$. The **offered bandwidth** measured in cache lines server per service time is:

![[Pasted image 20250518003842.png | 600]]

### Processing in Memory
The **memory wall** is the [[Von Neumann Bottleneck]] (not becauses memories are slow, but because communication with them is slow). An idea is to incorporate processing logic into the memory system (doing **vector instructions** on a bank/chip). 

The idea has got a renewed interest with 3D stacked memories. 

![[Pasted image 20250518004209.png]]

- **High bandwidth memory (HBM)** is a standard for 3D stacked memories widely adopted for GPUs (server solutions exist too)
- They provide high memory bandwidth compared with 2D traditional solutions (hundreds of GiB/s)
- They also can incorporate a Logic Die for processing (HBM-PIM), especially for AI/ML workloads Extensions of the ISA to trigger and control PIM computations directly in memory
# References