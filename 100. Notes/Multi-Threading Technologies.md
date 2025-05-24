**Data time:** 15:16 - 24-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Shared Memory Systems]]

**Area**: [[Master's degree]]
# Multi-Threading Technologies

[[Pipeline Processors]] and [[Super-Scalar Processors]] were not enough. Both techniques improve performance. Pipelining overlaps the execution of different stages, allowing a new instruction to start at every cycle. Superscalar execution allows multiple instructions to be issued and executed in parallel at each cycle. But they can still have **Sequential Bottleneck**. 

In **Sequential Bottleneck**  the number of independent instructions is small, and therefore, the amount of effective parallelism is low. **Stalls** (e.g., cache misses, data dependencies) can idle the pipeline if there aren’t enough ready-to-run instructions in a single thread.

To overcome such low efficiency, **HW Multi-Threading** has been added in superscalar processors to execute multiple instructions from multiple threads of control simultaneously. Each thread has its own set of registers and program counter (PC), representing its context. The processor maintains the context of each thread to quickly switch between them (Typically, 2-4 contexts per core).
### HW Multi-Threading
HW Multi-threading (or multithreading) enables a single core to execute multiple threads concurrently. A thread represents an independent sequence of instructions. There are three main types of HW multi-threading:
- **Fine-grained Multi-Threading (FMT)** interleaves instructions from different threads one at a time at the instruction level.
- **Coarse-grained Multi-Threading (CMT)**: switches between threads at the thread level. A switch happens only when the thread in execution causes a stall (e.g., L1 cache miss).
- **Simultaneous Multi-Threading (SMT)**: interleaves the execution of instructions from different threads. Instructions are simultaneously issued to the execution units from multiple threads at each clock cycle.

![[Pasted image 20250524153826.png]]

**Hyperthreading**: intel's specific implementation of **SMT**, which optimizes the use of resources (For the OS, each context is a logical core)

### Threading on [[Multicore Technologies|Multi-Cores]]
Parallelism is achieved by starting threads that run concurrently on the system. Thread creation is generally more lightweight and faster compared to process creation:
- Creating a process is roughly 3-5 times slower than creating a thread. In modern systems, creating a thread task takes $O(10⁴)$ **clock cycles**
- The fork system call requires copying/initializing more data (eg page table)

Data exchange among threads occurs by reading from and writing to shared memory. Processes data exchange usually requires IPC (i.e., system calls). Memory segment sharing also possible across processes (shared-memory).

### [[Data Race Problem]]

### [[False Sharing Problem]]
# References