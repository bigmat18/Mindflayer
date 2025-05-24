**Data time:** 15:48 - 24-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Shared Memory Systems]]

**Area**: [[Master's degree]]
# Data Race Problem

The **Data race (DR)** problem occurs when two (or more) threads access a shared variable simultaneously and at lest one access in a write operation, the accesses to the shared variable are not separated by synchronzation operation.

DRs produce **non-deterministic behavior** and debugging it is hard. [[Synchronization Basics|Syncronization mechanism prevent DRs]], only one thread modifies/reads at a time (mutexes, condition variables, semaphores, atomic instructions)

![[Pasted image 20250524155225.png]]

# References