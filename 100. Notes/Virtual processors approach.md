**Data time:** 14:11 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Virtual processors approach

This is a formal approach which, starting from the sequential, computation, is able to derive the basic characteristics of one or more equivalent data parallel computations. We use a methodology based on two **conceptual steps**:
##### Step 1
We identify an abstract representation of the computation by reasoning at the **highest parallelism** as possibile. The parallel entities of this step are called **Virtual Processing (VP)**

We will identify VPs by respecting the **Owner Compute Rule (OCR)**: each data element can be read by one more VPs but can be written by one VP only.

**Example with streams**:

![[Pasted image 20250514142817.png | 250]]
We unroll the *for* loop and we apply the [[Bernstein Conditions]] to check if iterations are independet
![[Pasted image 20250514142917.png]]
Iterations are indepent, so the calculation time can theoretically be reduced from O(L) to O(1) with teh highest parallelism degree. We have L VPs identifies as:
$$VP_{i=0,\dots,L-1} = \{A[i], B[i]\}$$
We identify a **Map** if each VP reads only l**ocal data elements** assigned to it (as in the example before). After the first step we can start the second phase.
##### Step 2
VPs are mapped onto a **concreate implementation** with modules implemented by **precesses**. Each worker executes a **subset of the VPs**, and the input and output data structures are assigned to workers according to a **mapping strategy**.

![[Pasted image 20250514143411.png | 400]]

### [[Map Parallelization]]

### Matrix-Vector Product
The **matrix-vector product** is an important computational kernel in linear algebra. Several optimized libraries for multicores execute it in a very efficient manner (e.g., **BLAS**, **LaPack**) according to algorithms that exploit at best caches and multiple cores. Now we see an **example with streams**:

![[Pasted image 20250514151205.png | 500]]

- We are considering a **stramed version** of this computation
- We assume that we receive a stream of pairs (the first element is an array of size L, the second a matrix of size $L²$)
- For each input array A, P applies the **matrix-vector product** and the resulting array C is produced onto the output stream
###### Step 1
We analyze the two for loops to understand which groups of iterations can be executed in parallel (**[[Bernstein Conditions]]**)

![[Pasted image 20250514151703.png]]

**All the iteration**s of the **outermost loop** can be executed in parallel while the iterations in the innermost loop not:

![[Pasted image 20250514151802.png |300]]
- **Output dependencies** in the iterations of the innermost loop (to update C[i]
- Completion time from $O(L²)$ to $O(L)$
- We recognize $L²$ VPs defined as $VP_{i=0,\dots, L-1} = \{A[i][*], B[*], C[i]\}$
- The definition respects the **OCR**
###### Step 2
The actual implementation based on processes performs **data distribution, computation, output collection**. It can be derivad from the VP and a mapping strategy. Let us assume **linear mapping**

![[Pasted image 20250514152825.png|300]]   ![[Pasted image 20250514152835.png | 350]]

- According to the **VP definition** and **mapping strategy**, we identify the required data distribution and collection
- For each input pair {A, B} the matrix B is **scattered** to the workers (**by rows**), while the array A is multicast to them
- Each worker produces a partition of C
- A **gather process** builds the whole C and transmits it
- Two implementations on above. The separation of scattering and multicast collectives into two processes is useful when S+M is the bottleneck.

# References