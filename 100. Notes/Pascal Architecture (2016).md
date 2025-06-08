**Data time:** 13:25 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# Pascal Architecture (2016)

Global memory up to **16 GiB** in the **P100** model, adequate (at that time) for deep learning computations. Support for emerging **High-Bandwidth Memory (HBM2, 3D stacked)**. Example: **P80** reaches **240 GiB/s** memory bandwidth. New type of **NVLINK bus** (about 80 GiB/s) outperforming PCIe 3.0 (up to 16 GiB/s with 16 lanes). Useful to connect more GPUs efficiently

![[Pasted image 20250601133041.png]]

### SM
Each SM has **64 CUDA cores**. **16 DPUs** per SM, 8 LD/ST units. Bigger register file. Up to **2048 CUDA threads** resident simoultaneously (the same as previous models).

![[Pasted image 20250601133136.png]]

### NVLINK
Interconnection network to connect more GPUs. Two GPUs can be connected directly. More GPUs can be connected via **NVSwitch**

![[Pasted image 20250601133224.png | 500]]


# References