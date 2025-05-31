**Data time:** 13:13 - 30-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Execution Model]]

**Area**: [[Master's degree]]
# Control Divergence on NVIDIA

**Example 1**: Threads take different branches of a conditional
![[Pasted image 20250530131155.png|300]]


**Example 2**: Threads execute a different number of loop iterations
![[Pasted image 20250530131400.png | 300]]

### Latency Hiding
When a warp needs to wait for a **high latency operation**, another eligible warp is selected and scheduled for execution on the same cores of the SM. This allows for hiding latencies.

![[Pasted image 20250530132307.png | 500]]

Each instruction run by a warp takes several cycles to complete:
- **<100 cycles** for arithmetic instructions
- **>100 cycles** for accesses to global memory

This latency should be masked by executing different active warps in an interleaved manner (**hardware multi-threading** like in CPUs). To maximize throughput (i.e., the ultimate goal of GPU computing), it is pivotal that the **warp scheduler** always have **eligible warps** to be promoted as **active warps** at every clock cycle.

![[Pasted image 20250530132501.png]]

The figure above assumes an LDR latency of at most of 9 cycles (not realistic). Five active warps hide latency. Much more warps are needed in the reality.

###  Logical-Physical Mapping

![[Pasted image 20250530132852.png | 550]]

Different logical concepts of memory in CUDA mapped onto different hardware supports on the device:
- **Global memory** persists between kernel invocations. It is allocated in the device memory (off-chip GPU RAM). 
- **Local memory** (per-thread) is still in the device memory
- **Shared memory** is accessible by threads of the same block. It is allocated in specific banks within a SM (on-chip, 100x faster)
- **Registers** of the SM are assigned to blocks (and then to its threads)

![[Pasted image 20250530133034.png | 500]]

Figure below of a **Stream Multi-Processor** of the **FERMI** GPU. **Cores** are Stream Processors (**SPs**), they are like EUs. L1 cache is private of a SM. **Registers** are potentially used by all cores in the SM (often thousands). **Load/Store units** to interact with the off-chip device memory. Units for **special functions** (e.g., sin, cos, exp, …). **Scheduler** of warps (groups of threads running in SIMD mode). Instruction cache.

![[Pasted image 20250530134054.png | 250]]

### SIMT Model (pre-volta)
**Single Instruction Multiple Threads (SIMT)**. Threads of the same block are executed in groups of 32 threads (with contiguous identifiers) called **warps**. Warps are executed on the same number (32) of cores of the SM assigned to the corresponding block

In **pre-Volta GPUs**, each warp has a **single program counter (PC)**, a **stack** shared by all the 32 threads, and an active mask telling which threads are currently **active** in the warp.

![[Pasted image 20250530134253.png]]

In case CUDA threads in a warp diverge, alternative code paths are executed at the **granularity** of **code blocks**, e.g., {A;B;} and {X;Y;} Divergent threads will reach the reconvergence point before switching

### Stack-based Re-convergence
Upon a divergent branch, threads in a warp are allowed to follow different **control-flow paths**. Different paths are serialized (**SIMD**) while the hardware tries to reconverge threads ASAP. The **immediate post-dominator point (IPP)** is the earliest point where all threads diverging at the branch are guaranteed to re-converge and execute again together

![[Pasted image 20250531100723.png | 550]]

On the right-hand side, the **control-flow graph** where D is the **IPP** of the branch taken with the if statement after A.

GPU hardware keeps a **per-warp stack** whose entries contain the following information:
- A **Program Counter (PC)**: with the address of the next instruction to execute
- An **Active Mask** indicates which threads have **diverged** in this path and which ones are instead **coalesced** and run the next instruction
- A **Re-convergence Program Counter (RPC)**: indicates which threads have diverged in this path and which ones are instead coalesced and run the next instruction

![[Pasted image 20250531101040.png | 400]]

- When we execute the branch (**if**), the entry of the stack changes by replacing the **PC** with the **IPP** of the **if**
- Two new stack entries are pushed: one for **branch-taken threads** and the other for **non-branch-taken ones**
- These two entries have their RPC equal to the IPP
- Only threads at the top of the stack (**TOS**) effectively run
- When threads reach the **IIP**, the entry of the stack is popped out

### SIMT Model (post-volta)
To enable fine-grain synchronization between parallel threads in a program, Volta GPUs (2017) introduced a new feature called **Independent Thread Scheduling (ITS)**. Threads of the same warp can be at different points since they now have their **PC**. Threads are still executed in SIMD mode (one instruction at a time), but the scheduler tries to advance warps in all possible existing divergent paths.

![[Pasted image 20250531101906.png]]

Both branches (**if** block, **else** block) will be executed in an interleaved manner. This is possible because every thread as a PC. A side effect of this scheduling is that threads might stay diverged until the program finishes. Re-convergence can be explicitly requested with a synchronization command: `__syncwarp`()

### Avoid Warp Divergence
There are situations in which, through a **static analysis** of the code, it is possible to **avoid thread divergence** by refactoring the code of our kernels. 
###### Example
![[Pasted image 20250531111836.png | 500]]

Standard solution with L Virtual Processors where:
$$
VP_{i=0\dots L-1} \{C[i], A[i], B[i]\}
$$
One VP is a CUDA thread, so half of the threads in a warp perform $A[i] + B[i]$, the other half perfom $A[i] - B[i]$. Thread divergence happens in each warp due to the assignment between elements of A, B and C and threads.

It is more convenient to reason in terms of warps, so avoiding the direct relationship between thread UID and positions in the arrays. We can create a new **indexing** where threads in the same warp perform the same calculation and write in the right positions of the output array C.

![[Pasted image 20250531112248.png | 400]]

- Example with **warps of 4 threads** (for the sake of simplicity in the example of the figure aside)
- Arrays of **8 positions** each
- **Even positions** ($A[i]- B[i]$) are computed by threads in the **first** warp
- **Odd positions** ($A[i] - B[i]$) are computed by threads in the **second** warp
- **No thread divergence**

The CUDA kernel is the following:
```c
__global__ void kernel_test(int *A, int *B, int *C, int size)
{
	unsigned int tUID = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int wID = tUID / warpSize;
	if (wID % 2 == 0) {
		unsigned int pos = (wID * warpSize) + (tUID - wID * warpSize)*2;
		if (pos >= size)
			return;
		C[pos] = A[pos] + B[pos];
	}
	else {
		unsigned int pos = ((wID – 1) * warpSize + 1) + (tUID - wID * warpSize)*2;
		if (pos >= size)
			return;
		C[pos] = A[pos] - B[pos];
	}
}
```

Each thread computes it unique thread identifier (`tUID`). The kernel is a **1D grid** of **1D blocks**. Each thread computes the identifier of its warp (`wID`). The variable `warpSize` is built-in and always equal to 32. The local variable (allocated in a register) `pos` is the position where to read A and B and write C by thread `tUID`.

Some performance results:
![[Pasted image 20250531113355.png| 300]]

Small improvement (a few usecs) for the `no_divergence` version. Higher improvement in general (the example is trivial, with very few instructions in each divergent branch)

# References
