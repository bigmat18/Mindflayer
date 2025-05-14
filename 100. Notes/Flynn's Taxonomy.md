**Data time:** 12:12 - 19-03-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Classifying Parallel Architectures]] [[High Performance Computing]]

**Area**: [[Master's degree]]
# Flynn's Taxonomy

In 1966 Michael J. Flynn classified four different families of architectures based on the number of instruction streams and data streams. This is a classification based on the number of **instructions and data streams**.

![[Pasted image 20250321142632.png]]
### [[SISD (Single Instruction, Single Data)]]
Refers to the traditional von Neumann architecture where a single sequential processing element (PE) operates.
### [[SIMD (Single Instruction, Multiple Data)]]
It executes the same operation on multiple data items simultaneously.
### [[MISD (Multiple Instruction, Single Data)]]
It employs multiple PEs to execute different instructions on a single stream of data (rarely used).
### [[MIMD (Multiple Instruction, Multiple Data)]]
It uses multiple PEs to execute different instructions on different data streams.


![[Pasted image 20250514235401.png | 600]]

Except for MISD systems, which have been designed only as research prototypes (e.g., for fault tolerance), all other categories originate a specific branch of parallel architectures.

![[Pasted image 20250514235457.png]]
# References
[Lessons Slides](4-Classification.pdf)