**Data time:** 16:49 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# Dynamic Parallelism in CUDA

Standard CUDA programs are **flat**: i.e., they perform kernel launches and, for best performance, each kernel had to expose enough parallelism to efficiently use the GPU. **For-loop programs** benefit from this approach, while **irregular computations** and **nested parallelism** suffer a lot.

Dynamic parallelism, introduced with Kepler (CC 3.5, CUDA 5.0), allows kernels to launch other kernels and manage inter-kernel dependencies. Nice feature to model **recursive parallel computations**.

![[Pasted image 20250601165147.png | 500]]

Starting from CUDA 5.0, a **host program** can launch a **coarse-grained kernel**, which in turn can launch **finer-grained kernels** to do work where needed. This avoids unwanted computations while capturing all interesting details/ Example of an application use case that benefits from this approach (i.e., a fluid simulation using an adaptive mesh)

![[Pasted image 20250601165345.png | 400]]

Dynamic parallelism is useful for problems such as:
- algorithms using **hierarchical data structures**, such as **adaptive grids**;
- algorithms using **recursion**, where each level of recursion has parallelism, such as quicksort
- algorithms where work is naturally split into independent **batches**, where each batch involves complex parallel processing but cannot fully use a single GPU.

###### Example of DP
A **child CUDA kernel** can be called from a **parent CUDA kernel** that can optionally **synchronize** on the completion of that child CUDA kernel (otherwise they proceed asynchronously). The **parent CUDA kernel** can consume the output produced by the **child CUDA kerne**l, all without CPU involvement.

![[Pasted image 20250601165643.png]]

In the example above, every thread of the parent kernel launches a child kernel having 256 threads. The child kernel inherits from the parent kernel certain attributes and limits, such as the L1 cache/shared memory configuration and stack size.

### Balancing Load
There are applications where the **computational load** depends on the **data**. Therefore, there might be some threads with more load than others. Such heavy threads have longer execution times that slow down the whole kernel runtime.

![[Pasted image 20250601165910.png | 600]]

The toy example above shows a kernel where each thread invokes a different number of times the `process` functions. The load is likely **imbalanced**, and the kernel completion time is given by the one of the slowest thread in the grid. Load imalance is often a problem on GPU with a **flat kernel**.

Dynamic parallelism can be used exactly for this purpose (balancing the load **automatically** and at **runtime**)

![[Pasted image 20250601170139.png | 500]]

The host program launches a **parent kernel** with a sufficient number of threads (how many?). Each thread of the parent kernel launches a **child kernel** in charge of computing all the elements of `data` assigned to it. Each child kernel has the right number of threads to compute the assigned elements of `data` (how many?).

The version below consists of a first parent kernel (`parentKernel`) directly called by the host program with a number of threads equal to the number of elements of the arrays `starts/ends` (i.e., number of segments $m>0$)

```c
__global__ void parentKernel(int *starts, int *ends, float *data)
{
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int N = ends[i] - starts[i];
	childKernel<<<ceil(N / 128.0), 128>>>(data + starts[i], N);
}
__global__ void childKernel(float *data, int N)
{
	int j = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (j < N) {
		process(data[j]);
	}
}
```

Each thread of `parentKernel` launches a `childKernel` with a number of threads N > 0 at least equal to `ends[i]—starts[i]`. Useful to balance the load (elements of `starts` and `ends` are not know statically).

### Kernel Nesting and Memory
Parent and child kernels have two points of guaranteed **global memory consistency**:
- When the **child kernel** is launched by the **parent thread**, all memory operations performed by the **parent thread** before launching the **child kernel** are visible to the **child threads** when they start
- When the **child kernel** finishes, all memory operations performed by the **child kernel** are visible to the **parent thread** once the **parent thread** has synchronized with the completed **child kernel**.

Example of a [[Data Race Problem|race condition]] between a parent kernel and its child kernel:
```c
__device__ int v = 0;
__global__ void childKernel(void) {
	printf("v = %d\n", v);
}
__global__ void parentKernel(void) {
	v = 1;
	child_k <<<1,1>>> ();
	v = 2; // <-- RACE CONDITION
	cudaDeviceSynchronize();
}
```

After the `cudaDeviceSynchronize` by the parent, all locations of GMEM written by the `childKernel` are visible to the parent thread in `parentKernel`. **Nota bene**: shared memory locations and local memory locations cannot be passed to the child kernel.

#### Explicit Synchronization 
Example to understand **implicit** and **explicit synchronization** with CUDA dynamic parallelism:
- Threads in the child kernel are guaranteed to see the modifications performed by **thread 0** in the parent kernel s.t. `data[0]=0`
- However, due to the `__syncthreads` call in the parent kernel, all threads of the child kernel see the modifications performed on `data[0], data[1], …, data[255]` by the parent kernel threads
- When the child kernel is complete, **thread 0** in the parent kernel sees all the modifications performed by the child kernel threads
- Other threads in the parent kernel can see the modifications performed by the child kernel only after the second `__syncthread` primitive.

```c
__global__ void childKernel(int *data)
{
	data[threadIdx.x] = data[threadIdx.x] + 1;
}

__global__ void parentKernel (int *data)
{
	data[threadIdx.x] = threadIdx.x;
	__syncthreads();
	if (threadIdx.x == 0) {
		childKernel<<<1, 256>>>(data);
		cudaDeviceSynchronize();
	}
	__syncthreads();
}

void host_launch(int *data) {
	parentKernel<<<1, 256>>>(data);
}
```

So doing, we can allow **all** threads of the parent kernel to see the modifications produced by the child kernel.

#### Implicit Synchronization
A child kernel always completes before the parent kernel that launched it, even if there is no explicit synchronization. **Example**: if a given thread in **Grid A** (the one that launched **Grid B**) completes its execution before Grid B, it remains active until **Grid B** is not complete.

![[Pasted image 20250601171741.png | 550]]

### Nested Dependencies
![[Pasted image 20250601171803.png | 600]]

Version of our first kernel "Hello_World" with dynamic parallelism in CUDA.
```c
__global__ void nestedHelloWorld(const int iSize, int iDepth)
{
	int tid = threadIdx.x;
	printf("Recursion=%d: Hello World from thread %d block %d\n",
	iDepth, tid, blockIdx.x);
	// condition to stop recursive execution
	if (iSize == 1) {
		return;
	}
	// reduce block size to half
	int nthreads = iSize >> 1;
	// thread 0 launches child grid recursively
	if (tid == 0 && nthreads > 0) {
		nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
		printf("-------> nested execution depth: %d\n", iDepth);
	}
}

nestedHelloWorld<<<1, n>>>(n, 0); // launch the parent kernel in the host side
```

The compilation of a project with dynamic parallelism requires **separate compilation** of the host and the device code, and then linking of them in the final executable.

![[Pasted image 20250601171940.png | 600]]


# References