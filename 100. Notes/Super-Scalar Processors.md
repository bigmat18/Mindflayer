**Data time:** 18:18 - 16-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]]

**Area**: [[Master's degree]]
# Super-Scalar Processors

Modern CPUs are highly parallel (and quite complex), mixing **pipelining** and **superscalar** technologies. uperscalar CPUs were designed to **execute multiple instructions from a single process/thread simultaneously** to improve performance and CPU utilization. The processor **fetches multiple instructions concurrently in a single clock cycle**. 

**Executes them out-of-order** (i.e., as soon as operands are available) to keep high utilization of the execution units. Results are then re-ordered (through a reorder buffer – ROB) to ensure they are written back to the register file or memory in the correct program order. 

Finally, instructions are **committed (retired) in program order or discarded (flushed)** due to branch misprediction together with any dependent instructions.

Each stage is designed (with more combination resources) to process $n > 1$ **independent instructions** in parallel. Implication in the whole micro-architecture design. 

**Example**: n=2 is called **2-way superscalar processors**
![[Pasted image 20250516182025.png | 600]]

- **L1i** is able to read two instructions at consecutive address
- **Decode**: RF is able to read 4 different registers in the same clock cycle
- **Execute**: 2 ALUs doing operations between two operands each
- **L1d**: able to read or write at two different addresses is the same clock cycle
- **WriteBack**: RF can write two different registers in the same clock cycle.

### Hardware Multi-Threading
N-way superscalar processors with large n (8-16) are very inefficient. This because it's difficult to find in the same program up to n independent instructions (eg the compiler might fill a long instruction with several NOPs) 

A solution is called hardware multi-threading (in the picture below **Simultaneous Multi-Threading**)

![[Pasted image 20250516182708.png | 500]]

- Every clock cycle each stage executes two instructions (one of **thread 1** and the other of **thread 2**)
- Each **thread context** runs a different programs.

Replication of must of the architectural state components (program counter, register file one per thread)

![[Pasted image 20250516182909.png | 350]]

###### Interleaved
- Fine-grained
- The processor runs two or more thread contexts at a time 
- Switching thread at each clock cycle
- If a thread is blocked, it is skipped
###### Blocked
- Curse-grained
- Thread executed until an event causes a delay
- Effective on in-order processors
- Reduce pipeline stalls
###### Simultaneous (SMT)
- Instructions are simultaneously issued from multiple threads to the execution units of a super-scalar processo
- Try to run as many instructions as possible in parallel (of the same thread, as in super-scalar processors, and of different threads in the same clock cycle)

# References