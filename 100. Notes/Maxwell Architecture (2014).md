**Data time:** 13:21 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# Maxwell Architecture (2014)

Composed of six **GPU Processing Clusters (GPC)**, each composed of **4 SMMs** (the equivalent of SMX in Kepler). Each SMM includes **128** CUDA cores. More resources in each SMM. No substantial increase in number of cores.

![[Pasted image 20250601132257.png]]

### SMM
Each SMM with **4 sub-units**. Each sub-unit with **32 cores, 8 LD/ST units, Instruction Buffer**. **Shared memory of 96 KiB** for each SMM. **L1/Texture** shared by sub-groups of two sub-units. No **DPU** unit to support double-precision FP computations (main target was gaming and consumer users). **64-bit FP computations** are less efficient (design compromise to limit costs).

![[Pasted image 20250601132505.png]]
# References