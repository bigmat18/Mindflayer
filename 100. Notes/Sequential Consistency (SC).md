**Data time:** 14:07 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory Consistency]]

**Area**: [[Master's degree]]
# Sequential Consistency (SC)

It is a theoretical model designed by **Leslie Lamport** in 1979. A multiprocessor is **sequential consistent**     (**SC**) if the result of any execution **is the same as if** the instructions of all the PEs were executed **in some sequential order**, and the instructions of each PE appear **in the order specified by its program**.

So, the multiprocessor is implemented is such as way that:
- Each instruction by any PE is started and completed in the program order
- All PEs always see the **same global order** of memory instructions which is consistent across all the elements.

![[Pasted image 20250520141704.png]]

Each PE issues (and completes) in memory instructions in the **program order**. Each memory instruction is executed atomically, and then the memory **randomly switches** to execute a memory instruction from any other PE ([[MP IO Non-Deterministic|non-deterministc]])

###### Example 1
P1, P2, P3 and P4 access an integer x. P1 and P2 change the value of **x** by writing **a** and **b** at different time instants. P3 and P4 read x at different time instants.
- **Fist case** (**admissible** with sequential consistency)
![[Pasted image 20250520142757.png | 450]]

- **Second case** (**not admissible** with sequential consistency)
![[Pasted image 20250520142809.png | 400]]

###### Example 2
Two processes P1 and P2 access a shared integer variable A and a shared Boolean Flag. Both are initialized to zero.

![[Pasted image 20250520143116.png | 300]]

At the end of the execution, the possible values of X and Y are:
- X=0 and Y=0, with total ordering **c;d;a;b**
- X=0 and Y=1, with total ordering **c;a;b;d**
- X=1 and Y=1, with total ordering **a;b;c;d**

**X=0 and Y=0** is not admissible because:
- We know that $a \to b$ and $c \to d$ by program order
- **X=0** implies $b\to c$ which implies $a \to d$
- $a \to d$ says **Y=1** which leads to a **contradiction**
 
### Cost of SC: HW Perspective 
The implementation of SC in modern machines prevents most of the optimizations that are possibile nowadays:
- **Write buffer** that re-order STORE instructions by the same PE
- **Out-of-order execution** of instructions
- **STORE atomicity** is costly (issuing PE waits for several ACKs)

##### Case 1: Hiding Write Memory Latencies
Overlap memory accesses with other independent instructions issued by the same PE. STORE are **asynchronous**, and pending STOREs are kept in a **write buffer** within the PE.

![[Pasted image 20250520144028.png | 500]]

Although correct from the viewpoint of the single program, the presence of a write buffer might violate Sequential Consistency.

##### Case 2: Out-of-order Processing
When an instruction is blocked (ess because suffers a **data hazard** or triggers a **cache miss**), other subsequent instructions might be ready to be executed.

**Example 1**
![[Pasted image 20250520144357.png | 450]]
When the control flow finds a conditional statement (if), we do not necessarily need to wait for the condition evaluation to proceed.

**Example 2**
![[Pasted image 20250520144606.png]]
**NB**: such reorderings respect dependencies within the same program run by the same process/thread. However, they break the conditions of Sequential Consistency.

##### Case 3: Breaking STORE atomically
**STORE atomically** is the property stating the existence of a total order of all STORE (SC provides STORE atomically). Lack of STORE atomicity can result in **non-casual executions**. **Casual execution** means the following: "if I see something and tell you, you will certainly see it too".

**Example**: processes P1 and P2 work on two shared variables A and FLAG both initialized to zero/false. A and FLAG are allocated in two different local memories (memories are still shared, ie it is a [[NUMA - Non Uniform Memory Access]])

![[Pasted image 20250520145625.png | 600]]

Firmware messages received in a different order because of [[MP IO Non-Deterministic|non-deterministic]] in the varius stages (ess in a network).

**Caches** do not help in solving the problem. Example with **three PEs** each one having its own **private cache**. The cache line of an integer variable A is initially in the caches of all the considered PE.

The figure below supposes that every time we change the cache line, the new value is propagated to all the other owners of the same line (ie **update-based mechanism**)

![[Pasted image 20250520150034.png]]

### Implementing SC
**Store atomicity** is required by SC (writes to different memory locations must be observed in the same order by all PEs). Each PE cannot start a memory instruction until the previous one completes. So, each PE has at most one outstanding memory instruction at a time. 

What does it mean for a memory access to complete?
- Case 1: **Memory read (LOAD)**. A LOAD is completed when its value is returned back to the issuing PE (ie, it is copied in the destination register of the instruction)
- Case 2: **Memory write (STORE)**. A STORE is complete when the written value is **visible** to all processors. **Visible** does not mean that the updated value has been seen by all PEs. It means that it is available to them (if they perform a subsequent LOAD instruction, the right value will be returned back)

Sequential consistency restricts several firmware and software (ie. by the [[Level-based view for parallel computing|compiler]]) optimizations. Therefore the model is very rigid and not efficient.

##### Benchmarks
Processors issue memory accesses one-at-time and stalls for completion.

![[Pasted image 20250520152704.png | 350]]

From 17-40% of the time (depending on the benchmark), processors are idle waiting for the completion of instructions. Low system utilization (**low efficiency**) even with caching.
# References