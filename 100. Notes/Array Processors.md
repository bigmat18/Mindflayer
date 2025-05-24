**Data time:** 15:04 - 22-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[SIMD on CPU]]

**Area**: [[Master's degree]]
# Array Processors

Each instruction is fetched from the memory and scheduled to all the **Execution Units (EUs)** by the same **Control Unit (CU)**. Each EU accesses data using the **Data Memory** through an [[Introduction to Interconnection Networks|Interconnection Network]] (EUs might also have **local caches**). Arrays processors are not versatile and they have been superseded by [[Graphical Processing Units (GPU)]].

![[Pasted image 20250522150958.png]]

### Connection Machine
The **Connection Machine (CM)** is a member of a serie of parallel supercomputers developed at MIT in early 1980. Each CM-1 microprocessor had its own 4 kilobits of random-access memory (RAM), and the hypercube-based array of them was designed to perform the same operation on multiple data points simultaneously, i.e., to execute tasks in single instruction, multiple data ([[SIMD (Single Instruction, Multiple Data)]]) fashion

![[Pasted image 20250522151202.png | 500]]

### From VP to SIMD Instructions
Following an example of a [[Introduction to Data Parallelism|data parallelism]] computation
```c
float A[L], B[L]; // two arrays
... // initialization of A and B
for i = 0 to L – 1 do
	B[i] = sqrt(A[i]);
```

We can express the computation with the **[[Virtual processors approach|Virtual Processors]]** model that we studied in the course

```
VP[i|i=0…L-1]:: { float A[i], B[i];
				  B[i] = sqrt(A[i]); }

LOADF RA-base, Ri LSL #2, RFa
SQRT RFa, RFb
STOREF RB-base, Ri LSL #2, RFb
```

Abstract SIMD instructions (a SIMD instruction in this abstract model is equivalent to L scalar instructions in parallel)

```
1. LOADF RA-base, #0, RFa[0] || LOADF RA-base, #4, RFa[1] || … || LOADF RA-base, #(L-1)4, RFa[L-1]

2. SQRT RFa[0], RFb[0]|| SQRT RFa[1], RFb[1]|| … || SQRT RFa[L-1], RFb[L-1]

3. STOREF RB-base, #0, RFb[0] || STOREF RB-base, #4, RFb[1] || … || STOREF RB-base, #(L-1)4, RFb[L-1]
```

From `VP[L]` to $n_w > 0$ actual **Execution Units** `EU[0], ..., EU[n_w - 1]` each working on different elements of the array in parallel. It is a [[Map Parallelization|Map]], where each EU works on $\frac{L}{n_w}$ elements distributed to the EUs in a circuilar manner (**interleaved**).

![[Pasted image 20250522152552.png | 600]]

- Each instructions runs in parallel simultaneously in all EUs
- EUs running the same instruction access to consecutive memory addresses
- **Memory coalescing** can be used to merge all $n_w$ independent requests for distinct words into one memory request only, to save memory bandwidth. 

### Encoding of SIMD Instructions
The same encoding of scalar instructions for pipelined PEs with proper characterization for SIMD machines ess:
```
LOADF RA-base, Ri LSL #2, RFa
```
It is issued bu the Control Unit to all the EUs. Each EU has its registers `RA-base, Ri, RFa` and reads the referred values from its private caches or the memory. Register `Ri` is properly initialized in the EUs to load different elements of A in parallel.

![[Pasted image 20250522153341.png | 400]]

Each EU can maintain seveal copies of the registers for **hardware multithreading** like is SISD and MIMD machines.

**Example** of a SIMD program run by an Array Processor:
$$
\forall i = 0, \dots, L - 1 \::\: C[i] = sqrt(A[i])
$$
![[Pasted image 20250522153801.png | 600]]

Each instruction is executed by all Execution Units:
1. Each $EU_h$ writes the integer constant \#h into its local register `Ri`
2. Each $EU_h$ reads the float at address `RA-base + Ri * 4` into its `RFa`
3. Each $EU_h$ computes the square root of `RFa` and writes the result into its local `RFc`
4. Each $EU_h$ writes `RFc` at local address `RC-base + Ri * 4` in memory
5. Each $EU_h$ increments `Ri` by a constant equal to the \# of EU ($n_w$)
6. Each $EU_h$ computes the predicate `Ri < RL`, if true, it waits the next instruction from the Control Unit, otherwise stops the execution

### Divergent Branches
Conditional statements with a **global predicate** do not introduce problems, all EUs compute the same predicate, conditional statements with a **local predicate** are critical since EUs can compute a different result and their control flow might take divergent paths.

The problem is called **Prediction**. Example:

![[Pasted image 20250522154605.png]]

- Instruction 3 is executed (predicate evaluation) by all EUs on their local registers
- Let `{EU_true}` and  `{EU_false}` be the subsets of EUs that satisfly or not satisfy the predicate (IF > 0 RFA), respectively
- Instructions 4 and 5 are issued by CU to all EUs, they are executed by `{EU_false}` only, while `{EU_true}` receive the instructions bur do not execute them, ie, such EUs are in an **idle state** during the execution of 4 and 5
- Instructions 6 is executed by `{EU_true}` only, while `{EU_false}` are in an **idle state**
- Instructions 6 is executed by all EUs

Contitional statements with local predicates impair the **[[Optimal Parallelism Degree|actual parallelism degree]]**, thus the preocessing bandwidth. When the "if-then...else" is long or contains **nested branches**, there is little or no benefit compared with scalar execution.
 
 An **Implementation Idea** is: each instruction is associated with its **program counter (PC)** value, used as a unique identifier.

![[Pasted image 20250522160928.png]]

- Instructions 3 = (3, IF > 0 RFa, 6)
- `{EU_true}`: all instructions with PC < 6 are received and ignored, until (6, SQRT ...) is received and executed
- `{EU_false}`: after (5, GOTO, 7), (6, SQRT ...) is received and ignored, (7, STOREF ...) is received and executed

The previous example is simplistic. In more general cases, with nested loops and arbitrarily long branches, the implementation complexity is higher.

### Array Co-Processors
**Array Processors** are fast for **[[Data Parallelism]] programs** but not for all programs (not general purpose indeed). Furthermore, they like **flat data structures** with contiguous memory layout.

**Example**: programs working on linked lists, graphs, irregular algorithms are not good candidates. For this reason, array proocessors were used as **co-processors** of general-purpose host machines. This has a price to pay:

![[Pasted image 20250522162247.png]]

- **I/O bandwidth and latency for data transfers between Host and the SIMD co-processor** are critical ‘Large’ pieces of code must be delegated to the SIMD machine, otherwise, the data transfer overhead dominates
- **This problem does not exist with SIMD vectorization on traditional PEs**, in which even a single vector instruction can be executed between scalar instructions in a conventional PE (however, vector parallelism is much more moderate, i.e., a few units compared with thousands)
# References