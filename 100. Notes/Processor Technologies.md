**Data time:** 17:15 - 16-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Shared Memory Systems]]

**Area**: [[Master's degree]]
# Processor Technologies

A **processing elements** (PE or core) consists of the processor unit(s), MMU(s) a local IO subsystem, wrapped units, private caches (L1i+d and L2). Following an **Abstract structure** of PE:

![[Pasted image 20250516173743.png | 500]]

The processor is responsible for interpreting assembler instructions (more precisely machine code instructions). Different **micro-architecture models** with different performance (**throughput**)/
#### [[Single-Cycled Processors]]
#### [[Multi-Cycled Processors]]
#### [[Pipeline Processors]]
#### [[Super-Scalar Processors]]

#### Communication Units
PEs usually have the possibility to generate short **firmware messages** to other processors (in the same CPU or in other CPUs) that are notified through the **interrupt mechanism**. We assume that each PE has a local I/O sub-system composed of at least one I/O unit called **UC (Communication Unit)**.

![[Pasted image 20250516191941.png | 500]]

Inter-processor communications can be useful to execute **kernel operations** (e.g., scheduling) or **synchronization** between processors (e.g., a sort of **event notification**).
# References