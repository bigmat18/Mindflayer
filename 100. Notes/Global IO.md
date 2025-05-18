**Data time:** 00:54 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory and Local IO]]

**Area**: [[Master's degree]]
# Global IO

Even **uni-processors** can be view as a [[NUMA - Non Uniform Memory Access]] because
- **I/O units** may have an **internal memory** (ora a few registers) that can be addessed by PEs using **Memory Mapped I/O (MMIO)**
- **I/O units** may have direct acces to the main memory of the system though **Direct Memory Access (DMA)**

The **Global I/O** is the sub-system for the interaction with external devices, [[Introduction to link layer|Network Interface Cards]], peripherals, and co-processors like GPUs and FPGAs

![[Pasted image 20250518005932.png]]
# References