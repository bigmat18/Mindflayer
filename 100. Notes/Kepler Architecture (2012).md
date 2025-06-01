**Data time:** 13:16 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# Kepler Architecture

**L2** is double the size of the L2 in [[Fermi Architecture (2010)|Fermi GPUs]]. The I/O-INF supports PCIe 3.0. GTS now called **Giga Thread Engine (GTE)**. SM called SMX (up to 16 instances), each having **192 cores** (3072 cores)

![[Pasted image 20250601131840.png]]

### SMX
Internal architecture of an SMX (Kepler GPUs). New **read-only cache** (for **texture** and **constant memories**). Bigger register file, double-precision FP units. More instruction **dispatchers** and **warp schedulers**. We have **2048 resident threads** per SMX, max **64 resident warps** per SMX

![[Pasted image 20250601132026.png]]

### Fermi vs Kepler
Table summarizing the main differences between Fermi and Kepler GPUs.

![[Pasted image 20250601132103.png | 600]]


# References