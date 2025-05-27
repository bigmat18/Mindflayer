**Data time:** 18:13 - 27-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Basics of Message Passing]]

**Area**: [[Master's degree]]
# Computation-to-Communication Overlap

Like in [[Communication Latency]]. A computing node is equipped with a [[Introduction to link layer|NIC (Network Interface Card)]] that sends data to and receives data from the network on behalf of the processor. The NIC can be a **specialized processor** (so-called **SmartNIC**) with multiple capabilities other than simple DMA (also processing, e.g., compression) 

The sender process may execute a **non-blocking send** to the destination process, the NIC executes the  data transfer and then notifies the sender process about the completion of the transmission. While the NIC executes the data transfer, the processor executes some other useful operations (e.g., it can execute, or partially execute, another task)

There could be full or partial overlap between the computation of tasks and the communication with 
other processes. **Such overlap is fundamental to mask (or partially mask) communication overhead**. 

The computing module C receives in input a list of tasks (**data stream**). For each input, it takes $T_{calc}$ time for the computation and  $T_{common}$ time for sending the task computed to the next module.

The **service time of a module** $C(T_S^{C})$ is the time interval between the start of processing two consecutive inputs. We hay have:
- $T_S^{C} = T_{calc} + T_{comm}$
- $T_S^{C} = \max(T_{calc}, T_{comm})$ 

![[Pasted image 20250527183229.png]]

Smart NIC may include **RDMA (Remote Direct Memory Access)** transfer. It enables memory-to-memory data transfer between two nodes without the continuous involvement of the OS.
- Infiniband provides hardware-level support for RDMA with dedicated switch fabrics
- ROCE (RDMA-over Converged Ethernet) implements RDMA on top of Ethernet, allowing the same low-latency advantages in data-center environments without requiring InfiniBand infrastructure

Hiding [[Communication Latency]] with RDMA. CPU offloads data transfer to the NIC and continue the processing
- Bypasses CPU to enable direct memory read/write operations to/from remote nodes
- Offers [[Zero-Copy Implementation]], kernel-bypass data transfer
- Common in high-performance networks (e.g., Infiniband)

RDMA API (e.g., libibverbs) are generally more complex than standard API (e.g., sockets). Some Message-Passing libraries (e.g., **MPI**) leverage RDMA under the hood for fast point-to-point data transfe
# Reference