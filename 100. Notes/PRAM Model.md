**Data time:** 17:39 - 26-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Models of Computation]]

**Area**: [[Master's degree]]
# PRAM Model

**PRAM** is the acronym for **Parallel Random Access Machine**. **Idealized Shared-Memory platform**:
- No [[Cache Memory]]
- No [[NUMA - Non Uniform Memory Access]] organization
- No [[Synchronization Basics]] overhead

![[Pasted image 20250526174733.png]]

- n processors $P_0, \dots, P_{n-1}$ are connected to a **global shared memory M**
- Any memory locations is uniformly accessible from any processor in **constant time**
- Communication between processors can be implemented by reading and writing to the globally accessible shared memory
- Synchronous execution model (**lockstep execution**)

 In PRAM model n identical processors $P_0, \dots, P_{n-1}$ **operate in lockstep**. In every step, each processor executes an instruction cycle in three phases:
 1. **Ream phase**: Each processor can simultaneously read a single data item from a (distinct) shared memory cell and store it in a local register.
 2. **Compute phase**: Each processor can perform a fundamental operation on its local data and store the result in a register.
 3. **Write phase**: Each processor can simultaneously write a data item to a shared memory cell, whereby the exclusive write PRAM variant allows writing only distinct cells while the concurrent write PRAM variant also allows processors to write to the same location (possible race conditions).

**Uniform complexity analysis**: each step on the PRAM takes O(1) time

Several types of **PRAM variants based on memory access** have been defined to solve conflicts that arise when processors read or write to the same shared memory location.
- **Exclusive Read Exclusive Write (EREW)**: No two processors are allowed to read or write to the same shared memory cell during any cycle
- **Concurrent Read Exclusive Write (CREW)**: Several processors may read data from the same shared memory cell simultaneously. Still, different processors are not allowed to write to the same shared memory cell
- **Concurrent Read Concurrent Write (CRCW)**: Both simultaneous reads and writes to the same shared memory cell are allowed. In case of a simultaneous write we further specify which value will actually be stored
	- **Priority CW:** Processors have been assigned distinct priorities, and the one with the highest priority succeeds in writing
	- **Arbitrary CW**: A randomly chosen processor succeeds in writing its value.
	- **Common CW**: If the values are all equal, then this common value is written, otherwise, the memory location is unchanged.
	- **Combining CW**: All values to be written are combined into a single value by means of an associative binary operation (e.g. sum, product, minimum, logical OR/AND)

![[Pasted image 20250526175434.png]]        ![[Pasted image 20250526175446.png]]

Complexity analysis for PRAM algorithms involves two measures: 
- **Time complexity (T(n))**: It considers the number of synchronous steps needed to complete the algorithm
- **Processor complexity (P(n))**: It considers the number of processors used to execute the algorithm

The cost of the algorithm is:
$$
C(n) = T(n) \times P(n)
$$
Optimal algorithms aim for cost proportional to the best-known sequential algorithm.

### Parallel Prefix Computation on a PRAM
Binary **associate** operation $\circ$ on the set X is: $\circ: X \times X \to X$. For example addition, multiplication, minimum, string concat and boolean operations like AND/OR

Obtaining $S= \{s_0, \dots, s_{n-1}\}$ from $X = \{x_0, \dots, x_{n-1}\}$ is called **prefix computation**. Own goal is design a const-optimal PRAM algorithm, ie $C(n) = O(n)$. We use the **recursive doubling algorithm** using $p=n$ processors

```c
for (j = 0; j<n; j++) do_in_parallel             // each processor j
	reg_j = A[j];                                // copies one value to a local register
for (i = 0; i<ceil(log(n)); i++) do              // sequential outer loop
	for (j = pow(2, i); j<n; j++) do_in_parallel // each processor j
		reg_j += A[j - pow(2, i)];               // performs computation
		A[j] = reg_j;                            // writes result to shared memory
}
```

We have: $C(n) = T(n,p) \times p = O(\log n) \times n = O(n \times \log n)$

![[Pasted image 20250526180547.png]]

Each element of the final sequence represents the sum of all preceding elements in the original sequence. This algorithm is NOT cost-optimal when using p=n processors.

To reduce the cost C(n) we can reduce the number of processes used (i.e., P(n)). Let’s use $p = n/\log(n)$ processors and the following algorithm:
1. Partition the n input values into chunks of size log(n). Each processor computes local prefix sums of the values in one chunk in parallel (takes time O(log(n)))
2. Perform the old non-cost-optimal prefix sum algorithm on the $\frac{n}{\log{n}}$ partial results (takes time $O(\log(n/ \log(n)))$ )
3. Each processor adds the value computed in step 2 by its left neighbor to all values of its chunk (takes time $O(log(n))$ )

![[Pasted image 20250526181146.png]]

We have: $C(n) = T(n,p) \times p = O(\log n) \times O(n/\log n) = O(n)$ Thus, this **algorithm is cost-optimal**

```c
// Stage 1: each Processor i computes a local prefix sum
// of a subarray of size n/p = log(n) = k
for (i = 0; i<n / k; i++) do_in_parallel
	for (j = 1; j<k; j++) do
		A[i*k+j] += A[i*k+j-1];
		
// Stage 2: Prefix summation using only the rightmost value
// of each subarray (O(log(n/k)))
for (i = 0; i<log(n / k); i++) do
	for (j = pow(2, i); j<n / k; j++) do_in_parallel
		A[j*k-1] += A[(j-pow(2, i))*k-1];
		
// Stage 3: each Proc i adds the value computed in Step 2 by Proc i-1 to
// each subarray element except for the last one
for (i = 1; i<n / k; i++) do_in_parallel
	for (j = 0; j<k - 1; j++) do
		A[i*k+j] += A[i*k+j-1];
```

### Sparse array compaction on a PRAM
Assume you have a one-dimensional array A where most entries are zero. We can represent the array in a more memory-efficient way by only storing the values of the non-zero entries (in V ) and their corresponding coordinates (in C).

![[Pasted image 20250526181639.png]]

We can use a parallel prefix approach using $p = n/\log(n)$ processors
- We generate a temporary array (`tmp`) with `tmp[i] = 1 if A[i] != 0 and tmp[i] = 0 otherwise`. We then perform a parallel prefix summation on `tmp`. For each non-zero element of A, the respective value stored in `tmp` now contains the destination address for that element in V .
- We write the non-zero elements of A to V using the addresses generated by the parallel prefix summation. The respective coordinates can be written to C in a similar way.

![[Pasted image 20250526181822.png]]

### Limitations of the PRAM model
- Unrealistic memory access. It assumes that all processors have uniform, instantaneous access to a single shared memory.
- It assumes perfectly synchronized processors executing in lockstep. In practice achieving and maintaining such global synchronization introduce significant overhead
- It ignores communication costs
- Algorithms that are cost optimal in PRAM might not scale well when implemented on real system with limited processing resource

While the PRAM model provides a clean framework for designing parallel algorithms, its simplifying assumptions make it less representative of real-world systems, where memory hierarchy, communication delays, synchronization overheads, and limited resources play a crucial role.

# References