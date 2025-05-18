**Data time:** 17:56 - 16-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] 

**Area**: [[Master's degree]]
# Multi-Cycled Processors

In the [[Single-Cycled Processors]] approach, the length of the clock cycle can be quite large since all the different phases need to stabilize before **rising edge** of the clock. The length is given by a **critical path** of the most expensive instruction (that is the LOAD).

A **multi-cycled processor** adds **non-architectural registers** to separate the stages. However, each instruction is executed sequentially (ie the fetching of instruction i starts when instruction i+1 completes)

![[Pasted image 20250516180022.png]]
# References