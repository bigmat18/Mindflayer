**Data time:** 13:47 - 17-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Shared Memory Systems]]

**Area**: [[Master's degree]]
# Multicore Technologies

[[Super-Scalar Processors]] have a **limited [[Scalability]]** since the logic to support multiple hardware threads is expensive in terms of chip occupation. Designers are obliged to increase the **CPU frequency**. However, power requirements have a grown dramatically as **chip density** and **clock frequncy** have risen.

![[Pasted image 20250517135040.png]]

Replication of a **simple PE (core)** on the same chip. No need to increase too much **CPU frequency** since better performance is obtained through **parallel processing** (this transition did not come for free).

A CPU with multiple core can be multiple CMP (chip multi-processors) organization.

![[Pasted image 20250517135555.png | 400]]

1. This is organization of some first CPUs. It is still used in some **embedded multicores** (ess. in some **ARM** cores)
2. In this organization, there is no on-chip cache sharing. It is useb by some vendors for specific architectures with high parallelism (50-100 and more)
3. In this organization, the **L2** cache is **shared** amond more PEs while the L1 cahces are private. The **Intel Core  Duo** for example had this organization.
4. In this organization, the L1 and L2 caches are **private** while there is bigger **L3 shared cache**. This is the organization of modern multicores like intel and AMD modern CPUs.

### Abstract CMP
CMPs are multi-processors on a single chip (also called **multicore** or simply CPU). A **Core** is a synonym of PE. They are complex systems implemented on-chip. **Abstract overview** of a CMP with our **vendor-agnostic terminology**

![[Pasted image 20250517140452.png | 350]]

Simplified schema: in some modern CMPs there are **integrated hardware accelerators** (ess GPUs). In the figure above, no **shared cache (L3 modules)** that might be otherwise connected through the on-chip network.

- **Memory Interface Units (MIMFs)** are fir the interconnection with the off-chip external memory
- **External Interface Units (ECT-INFs)** are for the interconnection with other CMPs in case of a large multi-CMP system
- **I/O interface Units (I/O-INFs)** interface with global I/O

**Example**: Interl i7-990x
This is the block diagram of an Intel Mutlicore. We can see also the **terminology** used in a specific vendor technology. The role of our **MINFs** and **EXT-INFs+I/O-INFs** units is done by **DDR Memory Controllers** and **QuickPath Interconnect**

![[Pasted image 20250517151552.png | 400]]
 
### On-Chip Interconnects
Two extreme cases of **on-chip interconnection networks** are below:
![[Pasted image 20250517151639.png | 400]]

- **[[Buses]]** has one single medium connected to all nodes. Cheap, but no parallelism. So, **minimum [[Processing Bandwidth|bandwitdh]]** and **maximum [[Communication Latency|letency]]**

- **[[Crossbars]]** is a fully interconnected structure with $N²$ dedicated links. So **maximum [[Processing Bandwidth|bandwidth]]** and **minimum [[Communication Latency|latency]]**, suitable for limited parallelism only (ess. N=8) because of **link cost** and **pin-count** reasons.

**Limited degree networks**. Much lower cost than crossbars but with a smaller number of links at the expense of **latency** (ess. **logarithmic** is some notable cases) but **maximum bandwidth** can be achived.

The internal network of CMPs is used for connecting:
- **PEs to MINFs** for example in accesses to the external memory
- **PEs to I/O-INFs** for accessing the global I/O
- **PEs to PEs** for the cache coherence protocol and direct inter-processor communications
- **PEs to EXT-INFs** to communicate the other CMPs

Typical **limited-degree networks** used inside CMPs

![[Pasted image 20250517153842.png | 400]]
- **Switch units** doing routing and flow control
- **Rings** or **Multi-Rings** are used in CMPs with moderate parallelism
- **Meshes** are usually used for highly parallel CMPs for easier integration reasons (all links have the same length except in **toroidal meshes**)

##### Bus Arbitration
In the bus interconnection network, one link (a shared medium) is shared by multiple interconnected units. Only one unit at a time can write a message on the shared medium, while all the others can potentially read the same message (**snooping**).

An **Arbiter Unit** is in charge of grating access to the bus to a requesting unit. Different implementation schemes.

![[Pasted image 20250517153528.png | 550]]

### Single-CMP systems
Complex shared-memory systems can be built around a single multi-core CPU (CMP). An **[[Introduction to Interconnection Networks|external network]]** might exists between MINF units and the memory sub-system. In the figure below a simplified representations.

![[Pasted image 20250517154444.png | 500]]

##### Heterogeneity
A heterogeneous **System-on-Chip** (SoC), comprising a CPU and an integrated GPU (iGPU), can be used in a system with a **discrete GPU** (i.e., dGPU). **PCIe** is not a bus despite the name, but a very complex point- to-point interconnect SoC and dGPU have **separate physical memories**. dGPU can access the SoC’s memory through the PCIe interconnect.

![[Pasted image 20250517154722.png | 400]]

### Multi-CMP Systems
Large highly parallel shared-memory systems are usually equipped with **several multi-core CPUs** (CMPs), each incorporating a **local memory sub-system** accessible through MINFs, and off-chip network connecting CMPs.

![[Pasted image 20250517154912.png | 550]]
The figure above does not capture completely the heterogeneity of modern parallel system (no hardware accelerator is depicted in the picture)

##### Hardware Accelerators
Heterogeneous computing platforms equipped with different hardware accelerators visible as **advanced I/O co-processors**.

![[Pasted image 20250517155115.png | 350]]
- **GPU**: [[SIMT-on-GPU|SIMT]] (**Single-Intruction Multiple-Threads**) co-processors suitable to accelerate data parallel computations. Not only for graphic tasks, but also **general purpose** and fully programmable.
- **Smart-NIC**: No longer **ASIC-based boards** with rigid and non-extensible functionalities, but rather small **System-on-Chip devices** composed of a traditional **NIC** and a small **multicore CPU** (used to accelerate networking functionalities like check-summing, routing, compression, filtering, ...)
# References