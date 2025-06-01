**Data time:** 13:33 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# Volta Architecture (2017)

Major redesign of NVIDIA GPUs. Introduction of a new ISA. Twice the schedulers of previous models. More powerful SIMT model (Independent Thread Scheduling). New support for AI (**tensor accelerators**). Below a V100 GPU.

![[Pasted image 20250601133358.png]]

### SM
![[Pasted image 20250601133413.png | 600]]

### Tensor Cores
Each tensor core operates with 4x4 matrices performing (efficiently) the following computation $A \cdot B + C$

![[Pasted image 20250601133458.png | 600]]


# References