**Data time:** 13:00 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# NVIDIA Architectures Overview

NVIDIA GPUs are available in two main series of products. Consumer models are oriented towards gaming. Datacenter models instead are optimized for HPC workloads.

![[Pasted image 20250601130055.png | 550]]

### Compute Capability
In the NVIDIA jargon, the **compute capability (CC)** identifies the features set and the tecnical properties of a GPU device (both hardware and software). In more recent GPU architectures, the provided throughput becomes higher with more hardware components (SMEM, caches, number of SMs, CUDA cores, etc...). Evolution:
- **Tesla** → CC major version **1**
- **Fermi** → CC major version **2**
- **Kepler** → CC major version **3**
- **Maxwell** → CC major version **5**
- **Pascal** → CC major version **6**
- **Volta/Turing** → CC major version **7**
- **Ampere** → CC major version **8**
- **Ada Lovelance/Hoppe**r → CC major version **9**
- **Blackwell** → CC major version **10**

During the compilation phase, the user can specify the CC with proper flags to produce the right code for the target GPU.

Example of features enabled by specific devices having a given compute capability. Important milestones: **double precision support (Tesla)**, **page migration engine (Pascal)**, **independent thread scheduling (Volta)**

![[Pasted image 20250601130449.png]]

### PCIe Interconnect
Multicore CPUs and GPUs have their own I/O controllers (in our jargon of the first part of the course, **I/O interface units**). The interconnection between CPU(s) and GPU(s) is often done through the **PCIe interconnect**

![[Pasted image 20250601130628.png | 550]]

H2D and D2H data transfers can likely be the bottleneck of a computation exploiting GPU hardware. A **PCIe lane** is composed of 4 links (two pairs), one used for transmitting data one for receiving data. Each lane is a **full-duplex byte stream** transporting 8 bits. Depending on the PCIe version, we can have 1, 4 up to 16 lanes.

![[Pasted image 20250601130721.png | 550]]

Maximum bandwidth has evolved from a few GiB/s to tens of GiB/s in the last more powerful versions of the PCIe interconnect.

### [[Fermi Architecture]]

### [[Kepler Architecture]]

### [[Maxwell Architecture]]

### [[Pascal Architecture]]

### [[Volta Architecture]]

### [[Ampere and Hopper]]

### [[Multi-GPU Systems]]

# References