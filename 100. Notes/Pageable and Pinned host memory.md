**Data time:** 17:28 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Pageable and Pinned host memory

The execution of `cudaMemcpy` is by default synchronous, i.e., the host waits for the completion of the memory operation. Over the years, several enhancements and alternative options have been provided to CUDA developers: 
- **Pinned** (page-locked) **memory** for asynchronous copies
- **[[Zero-Copy Implementation|Zero-copy]]** with pinned **memory**
- **Unified virtual addressing** (UVA)
- **Unified memory**

Two desired goals (often contrasting): **better performance, better programmability**

![[Pasted image 20250601173025.png | 400]]

We saw `cudaMalloc` to dynamically allocate global memory. The primitive `cudaMemcpy` performs synchronous **H2D, D2H, D2D** memory copies (also **H2H**, why?). We can delete dynamically allocated global memory via the `cudaFree` primitive. How can we initialize global memory since it is allocated by the host? In fact, once we do a `cudaMalloc`, the device memory is allocated but not initialized. Two ways:
- We can initialize the allocated global memory with data present in the host memory and copied via `cudaMemcpy`
- We can use the primitive
```c
cudaError_t cudaMemset(void *devPtr, int value, size_t count)
```

The second option initializes the buffer to the same byte (last 8 bits of `value`). Allocations by the CUDA API are **aligned to 256 bytes**.

### Global Device Variables
So far, we have allocated global memory **dynamically**. However, it is possible to allocate global memory **statically** in **global scope**. We still declare variables with the specifier `__device__`.
```c
__device__ float d_array[128]; // global device array

int main() {
	…
	cudaMemcpyToSymbol(d_array, h_array, 128*sizeof(float));
}
```

The symbol `d_array` cannot be explicitly deference through `&` since it is a CUDA symbol. Analogously, the same approach is required to initialize **constant memory** as we have already seen.

```c
__const__ float c_array[128]; // constant device array

int main() {
	…
	cudaMemcpyToSymbol(c_array, h_array, 128*sizeof(float));
}
```

**Example** of a code using global device variables that are accessed through a kernel:
```c
#include<stdio.h>
__device__ int d_value[10];
int h_value[10];

__global__ void write_value()
{
	d_value[threadIdx.x] += threadIdx.x;
	printf(”Value GPU = %d\n", d_value[threadIdx.x]);
}
int main()
{
	for (int i = 0; i < 10; i++)
		h_value[i] = 10*i;
	cudaMemcpyToSymbol(d_value, h_value, sizeof(h_value));
	write_value<<<1, 10>>>();
	cudaDeviceSynchronize();
	
	cudaMemcpyFromSymbol(h_value, d_value, sizeof(h_value));
	for (int i = 0; i < 10; i++)
		printf(”Value CPU [%d] = %d\n", i, h_value[i]);
	return 0;
}
```

Global and constant variables cannot be explicitly deferenced via `&` by the host code. They are actually a symbol present in the GPU lookup table. Indeed, in the previous examples, we have always used a
global variable as a **symbol** not as an address. The host program can get the device address of a global/constant variable using a primitive of the CUDA API.
```c
__global__ int devCount; // global device variable
…
int main()
{
	…
	int *devptr = NULL;
	cudaGetSymbolAddress((void**) &devptr, devCount);
	cudaMemcpy(devptr, &count, sizeof(int), cudaMemcpyHostToDevice);
	…
}
```
Once obtained the device address of a device variable, we can use a standard `cudaMemcpy` to copy data from/to that buffer.

It is useful to summarize the possible allocations in CUDA memory of different statements shown in the table below

![[Pasted image 20250601175417.png | 500]]

**Automatic scalar variables** of kernels are allocated into **registers**. If they are **automatic arrays**, the compiler allocates them in **local memory** (in the off-chip device RAM). **Shared variables** always have **block-wise visibility**.

### Pinned Memory
Host virtual memory is generally **pageable**, so it can be subject to **page faults**. With a H2D data transfer, the **CUDA driver** is responsible to configure the **DMA** on the GPU board to transfer the required data from the host memory to the device memory (opposite for D2H) on a page basis.

DMA works with physical addresses. OS could accidentally page-out the data that is being read or written by the DMA.

![[Pasted image 20250601175927.png | 400]]

If the host memory provided with a `cudaMemcpy` primitive is pageable, the CUDA driver uses an internal **pinned** (i.e., **page locked**) **buffer** (it cannot be paged-out). Consequence is an **extra copy** that represents an additional overhead (plus the **synchronous semantics** of `cudaMemcpy`).

The host program can directly allocate pinned memory. **H2D** and **D2H data transfers** involving a host pinned memory are **faster** since **no extra copy** is performed by the CUDA driver. CUDA API to allocate host pinned memory:
```c
cudaError_t cudaMallocHost(void **devPtr, size_t count);
cudaError_t cudaFreeHost(void *devPtr);
```

We should avoid excessive use of pinned memory since this can be detrimental for the host performance. It might reduce the amount of memory available to the system for paging.

```c
// allocate the host memory
float *h_a = (float *) malloc(nbytes);
float *d_a;
cudaMalloc((float **) &d_a, nbytes);
cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
```

H2D an d D2H are **synchronous** to the host program. The CUDA driver copies the content of the source buffer into an **internal pinned buffer** of the driver. Then, it configures the DMA to transfer each page of the internal buffer.

```c
// allocate pinned (i.e., page-locked) host memory
cudaMallocHost((float **) &h_a, nbytes);
cudaMalloc((float **) &d_a, nbytes);
cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);
```

H2D an d D2H are still **synchronous** to the host program. The CUDA driver directly configures the DMA to transfer each page of the source buffer **since it has already been registered as a page-locked area**.

Execution of `pageable_test` (nsys profiler):

Execution of `pinned_test` (nsys profiler):

Results confirm the much higher e**xploited memory bandwidth** (H2D and D2H) due to the eliminated **additional copy** that we save by using directly host pinned memory.

### Asynchronous Data Transfers
In addition to avoid extra copies, the use of pinned memory allows **asynchronous H2D** and **D2H data transfers**. New API:
```c
cudaError_t cudaMemcpyAsync(void *dst, const void *src,
size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
```

Such a call is **asynchronous** with respect to the host, so the call may return before the copy is complete.
```c
// allocate pinned host memory
cudaMallocHost((float **) &h_a, nbytes);
cudaMalloc((float **) &d_a, nbytes);
cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
// <- here the CPU does useful work that will be overlapped with the copy
cudaDeviceSynchronize();
// <- at this point the copy is certainly complete
```
It is worth noting that asynchronous memory copies (i.e., with `cudaMemcpyAsync`) might become synchronous if they involve an host memory buffer that is not page-locked (pinned)

### Registered Pinned Memory
So far, we have allocated pinned memory from scratch with a proper primitive (`cudaMallocHost`). Sometimes, it might be useful to convert an already allocated memory buffer, which was in pageable memory, to become a pinned buffer. From [[Pascal Architecture (2016)|Pascal GPUs]] (we refer to this as Pascal+), this is possible with the following primitive:
```c
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
```

```c
// allocate pinned host memory
float *h_a = (float *) malloc(nbytes);
cudaHostRegister(h_a, nbytes)
cudaMalloc((float **) &d_a, nbytes);
cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
// <- here the CPU does useful work that will be overlapped with the copy
cudaDeviceSynchronize();
// <- at this point the copy is certainly complete
```

### Zero-copy Memory
Since pinned memory is page-locked, **can the GPU access directly such buffers** (i.e., through the PCIe interconnect)? Old GPU models (CUDA 2.0)
```c
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
```

The buffer pointed by `pHost` must be previously allocated with `cudaHostAlloc` (it is similar to `cudaMallocHost` but takes additional flags; we need the `cudaHostAllocMapped` flag). The function returns a **device-accessible pointer** that points directly to the same pinned buffer in the host memory. Such utilization of pinned memory directly by the device is also called **zero-copy memory**.

```c
// allocate pinned host memory
float *h_a, *d_a;
cudaHostAlloc((float **) &h_a, nbytes, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_a, h_a, 0);
// <- the point d_a can be passed to a kernel for GPU processing
```

### Unified Virtual Addressing
With CUDA 4.0, **Unified Virtual Addressing (UVA)** has been introduced (before the host program and the GPU had different logical addressing spaces). The application has now a single virtual addressing space. 

**Practical consequences:** the device can access the pinned memory of the host using host pointers; `cudaMemcpy` primitives can be invoked without specifying where exactly the input and output parameters reside (the CUDA driver infers this).

![[Pasted image 20250601181953.png | 550]]

For example, with UVA the following snipped is totally correct:
```c
__global__ void print_kernel(float *elem)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx == 0) {
		printf("Value of float in pinned memory: %f\n", *elem);
	}
}

int main(int argc, char **argv)
{
	float *h_a;
	cudaMallocHost(&h_a, sizeof(float));
	*h_a = 10.725;
	print_kernel<<<1, 16>>>(h_a);
	cudaDeviceSynchronize();
	cudaFreeHost(h_a);
	return 0;
}
```

In a UVA regime, all pinned allocations are automatically mapped. No need to get a valid device pointer as before. Reading is done through the **PCIe** (not so fast, be careful if you have to read the same data repeatedly).
# References