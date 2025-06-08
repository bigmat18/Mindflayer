**Data time:** 13:37 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# Multi-GPU Systems

Multi-GPU systems comprise several GPUs connected with a multi-CMP host machine. GPUs communicate with each other in addition with the host. **Example**: NUMA of SMPs (two CPUs) connected with up to eight H100 GPUs (i.e., a very **large scale-up server**)

![[Pasted image 20250601133917.png | 500]]

**Traditional solution** (cheaper), with the two CPUs connected through the PCI interconnect to the 8 GPUs. Four pairs of GPUs are directly connected through the **[[Pascal Architecture (2016)|NVLINK interface]]** ($10² - 10³ GiB / s$), while GPUs in different pairs should pass through the **PCI** ($10⁰ - 10¹ GiB /s$).

In addition to being shipped as PCIe expansion boards, NVIDIA GPUs can be available as **SXM modules**. SXM is a socket for connecting GPUs with the rest of the system. **Example**: NUMA of SMPs (two CPUs) with up to eight H100 GPUs.

![[Pasted image 20250601134235.png | 500]]

**Advanced solution** (expensive), with the **two CPUs** connected through to the **8 GPUs** through the PCIe interconnect (more parallel than before) GPUs are available as **SXM modules**, capable of communicating throught four **NVLINK Switche**s in a all-to-all fashion (**[[Fully Connected|fully-connected network]]**)

### GPU Cluster
More scale-up servers (i.e., with two CPUs and up to 8 GPUs) can compose a **distributed architecture** called **GPU cluster**. GPUs communicate through the NVLINK within the same node. GPUs also communicate between nodes (**RDMA**).

![[Pasted image 20250601134430.png]]

**Intra-node structure:** 2 CPUs, two NICs for the host CPUs, 8 GPUs, 4 PCIe Switch Chips, 6 NVLINK Switch Chips, **8 dedicated NICs for the GPUs**. GPUs within the node can communicate with each other (i.e., GPU A accesses directly the memory of GPU B) through the **NVLINK interconnect**. GPUs in different nodes can communicate through their dedicated NICs (Infiniband based, e.g., 100 Gbps – almost 12.5 GiB/s). Technology **ConnectX-7** by NVIDIA
# References