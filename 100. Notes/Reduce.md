**Data time:** 22:37 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Reduce

Reduction is a very important parallel operator. It is a **second-order function** applied over the elements of a vector with an associative binary operator.
$$z = reduce(A, \oplus) = A[0] \oplus A[1] \dots \oplus A[L-1]$$
An **example** is the sum of all the elements of an array. The sequential implementation has a completion time O(L). However, this trivial version cannot be directly parallelized.

![[Pasted image 20250514224205.png | 400]]
According to the **[[Bernstein Conditions]]**, we have **read-after-write** data dependencies between iterations of the for loop.

We can rewrite the **reduction** in the following way owing to the **associativity** of the binary operator:

![[Pasted image 20250514224421.png | 500]]
We identify L Virtual Processors each one encapsulating one element of the array A. Completion time is reduced from O(L) (of the original algorithm in the previous slide) to $O(\log_2 L)$. The overall computation consists of $\log_2 L$ steps. At each step Virtual Processors communicate according to a different [[Stencil|stencil pattern]] (stencil is static and variable). Due to if statement in the **for** loop, only a subset of the VPs apply the operator $\oplus$ on two elements.

**Example**: Do a focus on the steps in case with L=8
- **Step h=1** communications between $VP_{i-1} \to VP_i, i = \{1,3,5,7\}$
![[Pasted image 20250514225137.png | 450]]

- **Step h=2** communications between $VP_{i-2} \to VP_i, i = \{3,7\}$
![[Pasted image 20250514225336.png|450]]

- **Step h=3** communications between $VP_{i-4} \to VP_i,i = \{7\}$
![[Pasted image 20250514225442.png | 450]]

We have identified three kinds of Virtual Processors:
- **White** receive an element from another VP and update their internal value
- **Red** only send their internal value to another VP depending on the step
- **Black** are idle

At each step the number of active VP halves. We map VPs onto $n>0$ workers (each worker executes $L/n$ VPs). Example with L=8.

![[Pasted image 20250514225847.png | 600]]

### Asynchronous Reduce
Reduction is often applied together with [[Map Parallelization|Map]] or other Stencil-based computations. It can be applied synchronously or asynchronously depending on the semantics of the original problem. Example of a **map+reduce** on streams:

![[Pasted image 20250514230611.png | 400]]

Reduce must be applied after the map (RAW dependency), but the map can be applied on the next stream element while the reduce is still computed on the previous one. In a first implementation, the **reduce** phase can be implemented as a **[[Farm]]**.

![[Pasted image 20250514230728.png | 550]]

Alternatively, worker processes in the map can also be involved in the reduce computation:
- Each worker receives a partition of the array A and applies the **map** to produce a partition B
- Each worker computes the **reduce** on all the elements of its partition of B. The output is the **local reduce result**
- Workers exchange their local reduce results to compute the **global reduce result**

![[Pasted image 20250514230936.png | 600]]

### Synchronous Reduce
Map+Reduce combinations can be based on a different pattern called **synchronus reduce**

**Example**: where s is an internal state 
![[Pasted image 20250514231351.png | 450]]
This computation is **stateless** and cannot be parallelized correctly with a **farm**. To compute the array $B(i)$ based on the input array $A(i)$ (the i-th output/input arrays), we read the value of $s$ representing the reduce computed at the end of the processing of $A(i-1)$. We have write after read
##### First Implementation
![[Pasted image 20250514231804.png | 550]]

- Each input array A is scattered to the workers (worker compute map using s, compute the local reduce result)
- Each worker computes its partition (local partition on s) of the array B. Before doing this, it waits for the global reduce result of the previous stream element from R
- Each worker sends the **local reduce result** on its partition of B to R
- R computes the global reduce results from the n local results received.
- R return the gobal reduce and each work start to compute its partition of A with function F

##### Second Implementation
parallel computation of the global reduce result and parallel [[Multicast]]. The example in the figure is with 8 workers but the cost model can be generalized for any parallelism degree:

![[Pasted image 20250514232245.png | 650]]

In this case we use classic reduce pattern to calculate s and after we propagate s back and in this process we compute F for each partition of A.
##### Stencil+Reduce
The reduce computation is often used also with **stencil-based programs**. For example, the **5-point stencil kernel** can be executed for several iterations that cannot be know beforehand. The iterations stops when a **global condition** over all the elements of the matrix is achieved.

![[Pasted image 20250514232440.png | 350]]

The two for loops are executed until all the elements of the matrix become lower or equal than a **threshold** (the stopping condition should be reachable according to the kind of computation F performed). The stopping condition is a property that must be checked over all the elements of the matrix (not only in one partition)
# References