**Data time:** 11:06 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Concurrency and Streams]]

**Area**: [[Master's degree]]
# Concurrency in CUDA

It is important to distinguish different **concurrency degrees** when we deal with complex programs for **heterogeneous computing** platforms including **multi-core CPUs** and **NVIDIA GPUs**.
###### CPU/GPU Concurrency
Since the host CPU and the GPU are different devices, they can work in parallel and independently
###### Copies/Kernels Concurrency
Owing to the presence of DMA(s) on the device, data transfers might happen while the GPU is running kernels on its SMs.
###### Kernels Concurrency 
From CUDA 3.0, GPUs are capable of running up to 32 kernels simountaneously on their SMs (recent GPUs reach 128). Kernels can be launched by different host threads of a multi-threaded host program (or by different host programs too).
###### Grid-Level Concurrency
The same host thread can launch different GPU tasks (e.g., kernels, copies) that might proceed in parallel on the device through the use of **CUDA streams**.
###### Multi-GPU Concurrency
For extremely compute-intensive applications, multiple GPUs available on the same machine can cooperatively execute the computation by partitioning the workload in a load balanced fashion (if possible).

### Host-GPU Synchronization
###### On the **GPU Side**
- At the **warp level**, the execution is effectively synchronous and the same instruction is executed by all the 32 threads of the warp in a SIMD manner. In case of **thread divergence**, a partition of the threads are active while other taking divergent branches are **idle**
- **Different warps** can execute completely different programs, and can arbitarily overlap their running times (i.e., SIMT model is not only SIMD). Correctness can be enforced with primitives like `__synchthreads` and `__syncwarp`
- **Different blocks** run independently and their execution can be overlapped (partially or totally). Blocks that consume SM resources are called **resident**, while others are **waiting**
###### On the **Host Side**
- There are several CUDA calls that are **synchronous** with the host program (e.g., the `cudaMemcpy`)
- Kernel launches are always **asynchronous** with the host program, and explicit synchronization can be enforced with the use of the primitive `cudaDeviceSynchronize`
- There are CUDA calls that are **asynchronous** (e.g., `cudaMemcpyAsync`). Waiting the completion of an asynchronous primitive can still be done via `cudaDeviceSynchronize`.


# References