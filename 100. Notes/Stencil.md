**Data time:** 16:09 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Stencil

If we analysis the first step of [[Virtual processors approach]] we can recognize a **stencil-based computation** when exists some VP that need to read data elements owned by other VPs.

**5-point stencil**: It's a kernel that recurs in many scientific applications, notably in the numerical solutions of partition differential equations (finite difference methods).

![[Pasted image 20250514220554.png | 650]]

- **Stopping condition**: For now, we assume a fixed number of iterations I>0
- **Ghost cells**: elements in the first/last row/column are not updated (**boundary condition**)
- **Swapping**: between A and oldA can be done by **exchanging pointers** instead of doing a costly plain copy of the matrix.

According ti the **[[Bernstein Conditions]]**, all iterations of the two innermost for loops are independent, so we recognize $L²$ VPs that reduce the [[Completion Time]] for each iteration from $O(L²)\to O(1)$.
$$VP_{i=0 \dots L-1, j=0\dots L-1} = \{oldA[i][j], A[i][j]\}$$
We have a **stentil** because each VP needs to read four neighbor elements owned by other VPs.

![[Pasted image 20250514221354.png | 450]]

### Row-wise Mapping
We have to decide on a mapping strategy. The **actual stencil** pattern (between proccess) might change (or can be even nullified) compared with the one between VPs. A fist strategy is called **row-wise mapping**

![[Pasted image 20250514222046.png]]

Each worker process has a set of contiguous rows of matrices oldA and A. Partition size is $L² / n$. Each worker process sends the **first row** and the **last row** of its partition to other two workers.
###### Implementation
![[Pasted image 20250514222237.png | 500]]

A similar approach is based on a **column-wise mapping**, where each worker has a contiguous set of $L / n$columns.
### Block-wise Mapping
Another, more interesting solution, is based on a block-wise mapping. Each worker sends the **boundary elements** of its block to four neighbor workers (the ones over the **perimeter** of its partition).

![[Pasted image 20250514222428.png | 500]]

### Asynchronous Stencils
In the previous implementations, the impact of the stencil is evident in the computation performed by the workers. They cannot start the computation until the communication with their neighbors is done. **Is it possible to mask this overhead?** Let us consider row-wise mapping.

![[Pasted image 20250514222702.png]]

### Floyd Warshall
The stencil pattern (both at the Vps level and at the processes level) can change during the iterations of the computation. An interesting example is the parallelization of the **Floyd-Warshall Algorithm** for computing the shortest path where work on the weight matrix.

Following the sequential algortihms:

![[Pasted image 20250514222848.png|350]]
- Iterations in the two innermost loops are fully independent
- Theoretical parallelism degree allows the completion time to be reduced from $O(L³) \to O(L)$

Each Virtual Processor executes L steps, and the communication pattern changes at every steps. To derive the stencil pattern, and how it varies at each step let us consider a small example with L=3

![[Pasted image 20250514223216.png | 600]]

Implementation with **row-wise mapping**. At step h, the worker with h-th row in its partition sends it to all workers.

![[Pasted image 20250514223311.png]]

For example, at the first step (h=0), the first row of the matrix held by the first worker is sent to all the other worker processes. In the second step, it is the same for the second row and so forth.

**Note**: red arrows above never generate **inter-process communications** with this mapping strategy.
# References