**Data time:** 23:32 - 22-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]]

**Area**: [[Master's degree]]
# Graphical Processing Units (GPU)

CPUs and GPUs follow a completely different design:

![[Pasted image 20250522233815.png]]
###### CPU
- A few powerful **ALUs**
- Sophisticated control
- Branch prediction to reduce control hazards
- Data forwarding to reduce data hazards
- Modest **mult-ithreading** to hide short latencies
- High clock frequency
###### GPU
- Many small **ALUs**
- Simple control
- **Long pipelines**
- Smaller caches
- More chip area dedicates to computation
- **Massive number of threads**

Comparizson of **peak memory bandwitdh** in GiB/s and **peak double precision GFLOP/s** for GPUs and CPUs since 2008. Data for Nvidia **Volta V100** and Intel **Cascade Lake** Xeon SP are used for 2019 and project into 2020. There is a huge difference (at least 10x but also more) owing to the completely different design of GPUs.

![[Pasted image 20250522234304.png | 500]]

GPUs were born to accelerate **graphical tasks** (ie processing a massive number of pixels by applying the same computation on all of them). Before 2007, the only way to program a GPU was by using **Graphics APIs** like OpenGL, Direct3D; but GPUs were also used to accelerate non-graphics workloads, however, computations had to be rewritten as functions over "pixels".

In 2007, NVIDIA released **CUDA**, it is a programming interface for using GPUs in a **general-purpose manner** (ie, not only for graphics workloads). This required an extension of the GPU architectural model.

GPUs are powerful for **[[Data Parallelism]] computations**. They provide a **huge parallelism degree** in terms of **cores** (order of thousands compared with 10-100 of CPUs)

### General-Purpose GPU Computing
As said before, **General-Purpose GPU computing (GP-GPU)** was enabled by CUDA for NVIDIA GPUs. The same for the other vendors (eg AMD) with OpenCL. The idea is to use GPUs for scientific applications, multimedia, engineering and many others not related to the graphics domain.

A **heterogeneous programming model** is needed. Developers design programs running both on the CPU and on the GPU.

**Separation of concerns**: the sequential part of the application is run on the CPU, and the parallel on the GPU. This is not always true: the CPU can be involved also in the parallel part too. The question is: how to develop code running on the CPU and how to develop code on the GPU? Same programming model? Different programming models? How to access data from the CPU and the GPU?

### Heterogeneous Applications
Parallel computing architectures are nowadays **heterogeneous**: i.e., they are equipped with more CMP-based CPUs and hardware accelerators. GPUs are **co-processors** and not standalone platforms (why? They are not good for all programs). The CPU(s) is(are) conventionally called the **host** while the GPU is the **device** in our jargon. [[Introduction to Interconnection Networks|Interconnection Networks]] between the CPU (or more CPUs) and the GPU (typically it is the **PCIe interconnect**)

![[Pasted image 20250523000005.png | 450]]
### Heterogeneous Applications
Heterogeneous applications are characterized by
- **Host code** running on the CPU (it is sequential in general, but it can also be parallel, so multi-threaded)
- **Device code** on the GPU (always highly data parallel).

The host is usually utilized to coordinate the GPU execution (data preparation, data copies, I/O, and managing the GPU runtime environment)

![[Pasted image 20250522235803.png | 450]]

### SIMT Paradigm
We hinted before that GP-GPU needs a new architectural model for GPUs that goes beyond the traditional distinction proposed by [[Flynn's Taxonomy]]. Compared with [[Array Processors]] that are purely SIMD, GPUs mix **[[SIMD (Single Instruction, Multiple Data)]]** and [[MIMD (Multiple Instruction, Multiple Data)]] with **hardware multi-threading**. This leads to a new hybrid paradigm called **Single Instruction Multiple Threads**

![[Pasted image 20250523000502.png]]

- In **SIMD**, there is one single instruction flow working on all the execution units
- In the case of divergent branches, only a fraction of the execution units work in parallel, while others remain idle (lowering efficiency)
- On GPU, with the **SIMT** model, a program (**kernel**) is a collection of **threads**
- Each thread has its **code, registers**, and **PC**
- Programmers develop their GPU program by writing the code of the threads in their kernels without taking care of the SIMD execution model (SIMD in not exposed to the programmer)

GPU is a collection of **GPU cores** (execution units actually, no control logic). A group of contiguous threads in the same kernel (usually 32) is a **wrap**. Each **wrap instruction** is scheduled by the GPU hardware on a group (usually 32) of GPU cores simultaneously. Threads in the same warp run in a SIMD (**lockstep**) manner.

![[Pasted image 20250523000902.png]]

Cores in the same 32-wide group run the same wrap-wide instruction at a time (**SIMD**). Cores in different groups run instructions of different wraps in parallel (**MIMD**). More wrap-wide instructions are scheduled in an interleaved manner to the same 32-wide group of cores to hide latencies, **hardware multi-threading** 
### NVIDIA Architectures
An NVIDIA GPU is a scalable set of **Stream Multi-Processors (SMs)**. Each SM supports the concurrent/parallel execution of hundreds of **threads**. Each SM runs threads in group of 32 (**wrap**): all threads of the same wrap execute the same instruction (SIMD). Each SM is composed of several **CUDA cores** (essentially it is an **EU** with **FP** and **INT** compabilities)

![[Pasted image 20250523001511.png]]

SMs groupped into multiple chips (**Graphics Processing Clusters**)

![[Pasted image 20250523001555.png | 450]]

##### GPU Evolution
A not so recent GPU (**Volta V100**) → 80 SMs, 60 cores (SPs) per SM, total 5120 cores. Special T**ensor Cores** (8 per SM) for accelerating floating-point operations.

![[Pasted image 20250523002601.png | 400]]

### Discrete CPU-GPU Systems
Traditional **Discrete GPU** architectural model (**dGPU**). Host machine (MIMD with several CMPs) and a GPU as a co-processor connected throught the **PCIe Interconnect** or using other proprietary interconnecting structures (eg **NVIDIA SXM**).

![[Pasted image 20250523002910.png | 500]]
### Integrated CPU-GPU Systems
Overcoming the latency and bandwidth problems of CPU-GPU interaction. GPUs as **on-chip co-processors** integrated on the same chip with the CPU (**iGPU**).

![[Pasted image 20250523003026.png | 500]]


# References