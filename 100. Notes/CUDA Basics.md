**Data time:** 00:30 - 23-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to CUDA]]

**Area**: [[Master's degree]]
# CUDA Basics

It enables **GP-GPU computing** on NVIDIA GPUs. Provides an **API** for GPU memory management, and for writing computational **kernels** and their companion functions to run on GPUs. API fully compliant with recent **C/C++** standards (ie **C++17**). GPU is viewed as a **compute device** acting as a co-processor for the host CPUs.

GPU has its own memory (called **Global Memory**) in the CUDA jargon. Basic programming concept is a **kernel** composed of thousands of **threads** (independent execution flows). Threads are grouped into **blocks** (each block runs on a SM of the GPU). Threads are executed in **warps** on the available cores of the chosen SM.
### CUDA Platform
The CUDA platform can be accessed through **CUDA-based high-level libraries, compiler directives** and **low-level application programming interfaces**. Support for doing that in **C/C++** but not only (**Fortan** too, also wrappers for **Python** and **Java**)

![[Pasted image 20250523132621.png|550]]
### CUDA APIs
CUDA provides an API organized in two main layers:
- **CUDA Runtime API**: they are high-level primitives written on top of the driver API
- **CUDA Driver API**: low-level primitives. They are hard to program but give full control on the device utilization and configuration

A CUDA program is composed of **host code** and **device code** running respectively on CPUs and on the GPU. The CUDA `nvcc` compiler separates host code from device code, and they take different compilation phases.

![[Pasted image 20250523132930.png | 550]]
  
### Compilation Workflow
Compilation of the host and the device part done integrally with `nvcc`

![[Pasted image 20250523133102.png | 450]]

In general, `.cpp` files contain CUDA primitives (eg device memory allocations) and **.cu** files kernels. We can also use **.cpp** file only. Device compilation produces a code written with a **virtual machine assembly (PTX)** independent from the specific GPU model. At runtime GPU model. At runtime, the PTX code can be compiled (**JIT**) to the actual executable for the target device.

The JIT overhead can be masked with JIT caching. Alternatively we can generate a **fat binary**.

 ![[Pasted image 20250523133425.png | 450]]

**Fat binary**: the compilation command includes some parameters indicating the specific physical model of the target GPUs. The output binary will include the physical machine code for all the specified target GPU models.

### Processing Steps
Copy of input data from the host memory to the GPU memory. 

- **Step 1**: The **GPU memory** can be addressed by the device only, the **host memory** can be addressed by any PE in every CMP (i.e., CPU) of the machine.

	![[Pasted image 20250528115211.png | 300]]

	Primitives to start the **H2D (host-to-device)** memory copy such that input data will be copied into the GPU memory. 

- **Step 2**: The host program loads the GPU program and triggers its execution on the device. During the execution of the GPU program, the c**ache hierarchy** of the device works to maintain in caches data that have been previously copied in the Global Memory by the host

	![[Pasted image 20250528115336.png | 400]]

	Primitives to launch a **CUDA kernel** on the device and to wait for its completion.

- **Step 3**: Once the kernel execution is complete (we need to be sure about this), output results have been materialized in proper buffers in the device memory. Copy the output results into the host memory to allow the host program to read and check them

	![[Pasted image 20250528115537.png | 400]]

	Primitives to start the **D2H (device-to-host**) memory copy such that output data will be copied into the host memory.

### [[CUDA Kernels]]


# References