**Data time:** 13:26 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory Consistency]]

**Area**: [[Master's degree]]
# Memory Consistency Basics

There is two kinds of information:
- **Private variables** for which the natural semantics is the **sequential one**, if we read the value we expect to see the latest value written in the same variable before.
- **Shared variables** can be simultaneously accessed by more process/threads. So, what do we expect by reading their values? We still expect to read the latest value written. What does the latest mean?

The answer to he last point is related to the **Memory Consistency Model** adopted by the architecture. The memory consistency model determines the order in which shared-memory accesses from different processes/threads can “appear” to execute. 

Memory consistency issues are often a matter of **system developers** of run-time supports for parallel programming. If synchronization mechanisms are properly implemented, **high-level parallel programmers** do not care about memory consistency models.

###### Example 1
Suppose that A and B are two memory locations. Each processor has its own set of registers numbered with unique identifiers **R0, R1, R2**. Suppose that A and B are both initialized to zero. Each processor has its own local cache and cache lines are kept **coherent** with each other through proper firmware mechanism and protocols

![[Pasted image 20250520134929.png | 300]]

All possible interleaving of instructions might be possible. Is it possible that at the end of the execution, both R1 and R2 are equal to zero? What would it mean if this outcome happens? This is possible in some machine and this is a counterintuitive behavior.

###### Example 2
Suppose a more complex scenario where we still have A and B, two memory locations shared by four processors this time. Suppose that A and B are both initialized to zero. Each processor has its own local cache and cache lines are kept **coherent** with each other through proper firmware mechanisms and protocols.

![[Pasted image 20250520140056.png | 500]]

We wonder if it is possible to have an execution outcome where **R1=0, R2=0, R3=1 and R4=0**. This would mean that Processor 3 and Processor 4 see the two STORE instructions performed by Processor 1 and Processor 2 in **different order**. This can happen in some machines, the order of store is different per process. 

###### Example 3
Suppose a different scenario where we still have A and B, two
memory locations shared by three processors. Suppose that A and B are both initialized to zero. Each processor has its own local cache and cache lines are kept **coherent** with each other through proper firmware mechanisms and protocols.

![[Pasted image 20250520140543.png | 400]]

We wonder if it is possible to have an execution outcome where **R2=1 and R3=0**. This would mean that Processor 3 sees the STORE on B by Processor 2, but when it reads A the STORE on A by Processor 1 is not visible yet. However, Processor 2 has already seen that STORE otherwise it could not have written 1 on B.
# References