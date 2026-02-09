**Data time:** 18:18 - 16-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Parallel and distributed systems. Paradigms and models]]

**Area**: [[Master's degree]]
# Super-Scalar Processors

Modern CPUs are highly parallel (and quite complex), mixing **pipelining** and **superscalar** technologies. superscalar CPUs were designed to **execute multiple instructions from a single process/thread simultaneously** to improve performance and CPU utilization. The processor **fetches multiple instructions concurrently in a single clock cycle**. 

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
# References