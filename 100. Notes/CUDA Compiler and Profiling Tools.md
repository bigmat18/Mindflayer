**Data time:** 16:18 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# CUDA Compiler and Profiling Tools

Source files include both **host** and **device code**. Good practice is to keep them in separated files (e.g., to reduce the compilation time). Code might contain **kernels, device functions, host functions, host+device functions**. The `nvcc` **frontend** separates host and device codes that are compiled separately. Device code is compiled first in a **PTX assembly**, tied to a specific **virtual architecture**, and then translated into a **cubin binary code**, tied to a target **real architecture**. Finally, the whole runnable program is generated in a **single executable**.

![[Pasted image 20250601162134.png | 400]]

The binary is executed by the host side. It includes some calls to the **CUDA runtime (RT)** that interact with the **CUDA driver**. The CUDA driver might compile (**JIT**) the `PTX` to a right cubin format supported by the target GPU (if the right cubin has not incorporated in the binary file yet). The **GPU driver** loads the program on the GPU memory, assigns blocks to SM, and performs the startup of the kernel.

![[Pasted image 20250601162320.png | 450]]

Therefore, the compilation process to generate the device code has two main stages (**alternative** or **complementary**)

![[Pasted image 20250601162413.png | 600]]

Why? Because NVIDIA wants to be able to push innovations on their hardware as soon as possible. They do not ensure **forward compatibility** of **cubin binaries**, unlike CPU vendors. They also break **backward compatibility**, since a new **cubiun binary** code cannot run on old GPU models (of course).

##### Real Architectures
Each generation of NVIDIA GPUs provides additional features and new functionalities. Also the **Instruction-set-Architecture (ISA)**, and the encoding of machine instructions, have changed a lot over the years. Real architectures are named `sm_xy`, where `x` is the GPU generation number, and `y` the version within the generation:

![[Pasted image 20250601162633.png]]

##### Virtual Architectures
GPU compilation is performed via an intermediate representation, PTX, which can be considered a sort of assembler language for a **virtual GPU architecture**.

![[Pasted image 20250601162813.png]]

PTX assembly is **forward compatible** with newer architectures, but it is not **backward compatible**. It is always possible to compile the **PTX assembly** of an earlier version (like `compute_70`) to a cubin binary for the most recent architectures (like `sm_90`). This is how NVIDIA ensures that old code will still run on.newer hardware.

### PTX Compilation
The **CUDA driver** (`libcuda.so`) contains the **JIT PTX compiler** and is always **backward compatible** (this is what actually makes PTX forward compatible). This means that it can take PTX assembly code from an older version and compile it for the current version of the device on the current machine. However, it is not forward compatible: code compiled with newer PTX assembly cannot be understood.

![[Pasted image 20250601163125.png | 300]]

If **driver 1.0** can understand the PTX version in our executable, it is able to generate the **cubin** for the target GPU that we have. This also works for any version of the driver >1.0 If we compiled the code with a PTX version that cannot be understood by driver X, we cannot compile it to a valid cubin with any driver \<X.

It is important to chose the right compilation flags. If our GPU is a **GeForce RTX 3090**, if we use `–sm_80` the code is correct but better performance might be achieved using `–sm_86`. **Example**

```
nvcc -x cu -O3 --gpu-architecture=compute_80 --gpu-code=sm_80 file.cpp
```

The option `--gpu-architecture (-arch)` is used to specify the virtual architecture, while the option `--gpu-code (-code)` to specify one or more real architectures. PTX assembly generated for a virtual architecture is always **forward compatible** with newer real architectures (not backward compatible).

This is useful to make a code portable on newer GPUs. Example of compilation (`PTX` of **[[Ampere and Hopper|Ampere]]** virtual architectures, cubin for **[[Ampere and Hopper|Hopper]]** physical GPUs).

```
nvcc -x cu -O3 --gpu-architecture=compute_80 --gpu-code=sm_90 file.cpp
```

If in the option -`-gpu-code (-code)` the user specifies a **virtual architecture** instead of a real architecture, the translation between the `PTX` code and the right **cubin** is deferred at runtime. The binary incorporates the PTX version. Example.
```
nvcc -x cu -O3 --gpu-architecture=compute_80 --gpu-code=compute_80 file.cpp
```

A different solution is the following:
```
nvcc -x cu -O3 --gpu-architecture=compute_50 --gpu-code=compute_50,sm50,sm52 file.cpp
```
We generate exact code for two [[Maxwell Architecture (2014)|Maxwell]] variants, plus `PTX` code for use by JIT in case a next-generation GPU is encountered. However, although forward compatible with new models, the command above generates a `PTX` of the [[Maxwell Architecture (2014)|Maxwell]] virtual architecture, with some recent features not available

### NVIDIA Profilers
NVIDIA provides a set of **profiling tools** that can be used toinspect the performance of a GPU code and understand potential **bottlenecks** both in the host and the GPU sides.

![[Pasted image 20250601164619.png | 600]]

Several profilers provided by NVIDIA over the years, with different features and characteristics. 
- Old GPU models (up to V100):
	- `nvprof` (CLI)
	- `nvvp` (GUI)
- Modern GPU models
	- `ncu, ncu-ui` (**Nsight Compute**) provide both CLI/GUI
	- `nsys, nsys-ui` (**Nsight Systems**) provide both CLI/GUI

To enable an effective profiling, CUDA code should be compiled with the additional flag `–lineinfo` (the    `–G` option is for debugging purposes only)

Profiling is a multi-step procedure. First, we usually need to profile **kernel codes**. As a second step, we might need to profile the **host code** to understand whether the use of the CUDA primitives is effective and efficient (e.g., H2D and D2H memory transfers, memory allocation, and so forth).
# References