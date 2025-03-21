**Data time:** 14:36 - 21-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Classifying Parallel Architectures]]

**Area**: [[Master's degree]]
# MISD (Multiple Instruction, Single Data)

This is the strangest type of architecture. In this type all PEs/CPUs execute a **different instruction sequence** on a single data stream.

This type of system is **not general-purpose** and not commercially implemented. One application of this is in mission-critical systems for fault tolerance reasons:
- The same data is processed by multiple machines and the decision is taken considering (for example) the majority principle.
- Different algorithms are run on distinct processors using the same input data.

![[Pasted image 20250321154901.png]]
# References