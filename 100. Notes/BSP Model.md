**Data time:** 18:19 - 26-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Models of Computation]]

**Area**: [[Master's degree]]
# BSP Model

The **Bulk-Synchronous Parallel (BSP)** model of computation was proposed by Leslie G. Valiant as a
unified framework for the design, analysis, and programming of general-purpose parallel systems.

It can deliver both scalable performance and architecture independence. It is not just a theoretical model, it can also serve as a **reference paradigm for parallel programming**. It consists of three parts:
1. A collection of processor-memory components
2. A communication network that can deliver point-to-point messages among processors
3. A facility for global synchronization (**barrier**) of all processors

![[Pasted image 20250526182328.png]]

- $n$ processors $P_0, \dots, P_{n-1}$ each with its own memory (a distributed memory multiprocessor).
- The remote memory access time is **uniform** (the access time to all non-local memory locations is the same).
- Communication between processors happen via explicit messaging (message-passing model)

### Algorithm
A BSP algorithm consists of a sequence of **supersteps**. A superstep consists of: 
- **computation**, **communication**, or **computation and communication** steps (so-called **mixed superstep** in case both are executed)
	- A **computation superstep** focuses on local processing tasks (e.g., floating point operations). 
	- A **communication superstep** focuses on exchanging data among processes (e.g., data words transferring). 
	- A **mixed superstep** may include both local processing and data exchange within the same superstep
- a **global barrier synchronization** (bulk synchronization) to ensure that all computation and communication from the superstep have completed

![[Pasted image 20250526183102.png]]

The cost of a **computation superstep** is:
$$
T_{comp}(w) = w + l
$$
where we have:
- $w$ is the amount of work defined as the maximum number of operations performed in the superstep by any processor
- Processors with less work than $w$ ops must wait

The cost of a **mixed superstep** is
$$
T_{comp}(w) = w + h \cdot g + l
$$
The same 𝑙 measure is used for all types of superstep,
##### h-relations
An **h-relation** is a communication superstep in which **every processor sends and receives at most h data**
**words**. It is the **maximum** between the data words sent and received by a processor in a communication superstep,

![[Pasted image 20250526183226.png]]

The cost of an h-relation is:
$$
T(H) = h\cdot g + l
$$
where we have:
- $g$ (gap) is the per-word communication cost and 𝑙 (latency) is the global synchronization time. Both are usually expressed in FLOPS (i.e., the time is multiplied by the processor’s FLOP rate
- Note that $g$ and $l$ depend on the number of processors (p)
- $l > 0$ includes the costs of the **global synchronization** plus all fixed costs for ensuring that all data have arrived at the destination, and the start-up of the communications
- Approximate values for 𝑔 and l on a given parallel computer can be obtained by measuring execution times for a range of full h-relations, varying h and p. A  full h-relation is an h-relation where each processor sends and receives exactly h data words.

### Const of BSP Model
The cost of a BSP algorithm is expressed as an expression
$$
C_{BSP} = a + b \cdot g + c \cdot l
$$
The expression is obtained by adding the costs of all the algorithm’s supersteps.

For **Example** with p=4, g=4 FLOPS, and l=20 FLOPS
- In the first comp. superstep $P_1$ takes 60 FLOPS, while in the second one $P_0$ takes 80 FLOPS
- In the first comm. superstep $P_1$ sends/receives 10 words to/from all other processors
- In the second comm. superstep $P_0$ receives 10 words from all other processors

$$
C_{BSP} = (60 + l) + ((3 \cdot 10) \cdot g + l) + (80 + l) + ((3 \cdot 10) \cdot g + l) = 460 FLOPS 
$$
![[Pasted image 20250526184142.png]]

### Dot Product BSP Example
Given $a = [a_0, \dots, a_{n-1}]^T$ and $b = [b_0, \dots, b_{n-1}]^T$ the **dot product** is $\alpha = a^T \cdot b = \sum^{n-1}_0 a_i b_i$. Let's consider $p$ processors and a **cyclic distribution** of the two arrays a,b
$$
a_i, b_i \to p_{i \mod p} i\in [0,n[
$$
Each processor computes a local partial dot product. 3 supersteps, in the end all p processor will compute $\alpha$

![[Pasted image 20250527154105.png]]

![[Pasted image 20250527154128.png | 450]]

### GEMM with BSP
Assumption: the matrix size is so large that **they are distributed evenly over p processors**
- A,B,C are $N\times N$, and initially each processor holds $\frac{N²}{p}$ values of A and B (ie, a sub block of A and B of size $\frac{N}{\sqrt{p}} \times \frac{N}{\sqrt{p}}$)
- At the end of the computation, each processor will hold $\frac{N²}{p}$ values of C

BSP algorithms: **2 supersteps**
- **Communication superstep**: data exchange among all processors to transfer the sub-blocks from/to other processors. The total words communicated per processor is about $\frac{N²}{p} \times \sqrt{p} (h = g \cdot \frac{N²}{\sqrt{p}} + l)$
- **Computation superstep**: all processors compute locally the $\frac{N²}{p}$ values of C (the cost is about $\frac{(2 \cdot N) \cdot N²}{p} = \frac{2 \cdot N³}{p}$ FLOP)

The cost of the algorithm is:
$$
C_{BSP}(N,p) = \frac{2 \cdot N^3}{p} + g \cdot \frac{N²}{\sqrt{p}} + 2 \cdot l
$$
While the [[Speedup]] is:
$$
S(p) = \frac{T(1)}{T(p)} = \frac{2 \cdot N³}{\frac{2 \cdot N³}{p} + g \cdot \frac{N²}{\sqrt{p}} + 2 \cdot l}
$$
The asymptotyc efficiency as N increase is $\lim_{N\to \infty} E(p) = 1$
# References
[L. G. Valiant. “A bridging model for parallel computation”. Communications of the ACM, Volume 33, Issue 8, pp. 103-111, August 1990.](https://dl.acm.org/doi/10.1145/79173.79181)