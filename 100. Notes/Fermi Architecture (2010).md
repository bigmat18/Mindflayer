**Data time:** 13:12 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# Fermi Architecture

Figure below of the **F110** having 32 cores per SM and 16 SMs (total 512 cores). GDDR5 Memory Controller is our MINF, PCIe Host Inteface our I/O INF unit. **Giga Thread Scheduler (GTS)**: it assign blocks to the SMs in a round-robin fashion setting `gridDim`, `blockDIm`, `blockIdx`

![[Pasted image 20250601131503.png]]

### SM
Internal structure of a **SM (F110)**: register file, 32 cores, 4 special functional units, 16 LOAD/STORE units. 64 KiB internal memory used both as a **L1 cache** and/or as a programmable **shared memory** (respective sizes controlled with the CUDA API).

![[Pasted image 20250601131652.png]]


# References