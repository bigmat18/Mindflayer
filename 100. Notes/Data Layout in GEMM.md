**Data time:** 21:24 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Data Layout in GEMM

In addition to coalescing the accesses to GMEM, it is often important to organize our data in a proper manner. Distinction between two layouts:
- **Struct of Arrays (SoA)**
- **Array of Structs (AoS)**

![[Pasted image 20250601211913.png]]

The SoA layout makes full use of GPU **memory bandwidth** because there is no interleaving of elements of the same field. Furthermore, the SoA layout provides coalesced memory accesses and achieves **more efficient GMEM utilization**.

##### Example of AoS
Example with a kernel composed of a **1D grid** of **1D blocks** reading points in a 3D space and multiplying the integer data value associated with each point by a constant `alpha`. Data layout is **AoS** (contiguous sequence of AoS structures defined below)

![[Pasted image 20250601212103.png | 550]]

Each thread reads and writes the `data` element of its corresponding point of the input data structure. Contiguous threads of the same warp access a 32-bit value with a spacing of **16 bytes** (4x L1 transactions, **25%** efficiency, accesses are **not coalesced**)

##### Example of SoA
In this slide, we propose an alternative layout for the same basic problem, showing its superior performance on a GPU. We choose a **SoA** layout, with four arrays storing contiguously in GMEM the same field for each input element of the input data structure

![[Pasted image 20250601212252.png | 550]]

Each thread still accesses the data field of the corresponding input element. However, now data fields are stored contiguously in GMEM. Accesses by contiguous threads of the same warp are separated by **4 bytes** (1x transaction per warp, **coalesced**).

### Performance Results
Results confirm that the SoA layout is to be preferred one against the most intuitive AoS. Results with an input of size $2^{10}$ **elements** and an alpha value of 25.

|     | Kernel running time<br>(usec) | Benchmark running<br>time (usec |
| --- | ----------------------------- | ------------------------------- |
| Aos | 0.61                          | 7385,93                         |
| SoA | 237.28                        | 6728,20                         |
Significant improvement with a reduction of the completion time (of the kernel) of **37%**. This example shows the importance of preparing the **data layout** for GPU processing in a careful manner. The choice of the layout depends on the type of computation performed and not always the layout optimizing a kernel will optimize other kernels working on the same data.


# References