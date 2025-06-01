**Data time:** 18:25 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Unified Memory (UM)

[[Pageable and Pinned host memory|UVA]] allows a single virtual addressing space. The same pointer is always meaningful and the pointed data can be physically in the host memory or in the GPU memory.

**Unified Memory** (CUDA 6.0 with [[Kepler Architecture (2012)|Kepler GPUs]]) extends UVA with **automatic data movements** between host/device memories during the access to a data. Achived goals by UM:
1. improve code productivity by making CUDA code easier to implement
2. data access speed may be maximized by migrating data towards entities that access it most frequently
3. total system memory usage may be reduced by avoiding duplicating memory on both CPUs and GPUs
4. it enables GPU programs to work on data that exceeds the GPU memory’s capacity
5. it is possible to eliminate the use of `cudaMemcpy` and analogous primitives, thus making the code easier to understand and maintain

Unified memory buffers are also called **managed memory** in CUDA. Dynamic allocation of managed memory:
```c
cudaError_t cudaMallocManaged(void **ptr, size_t size, unsigned int flags);
```
Flags can be `cudaMemAttachHost` (managed memory accessible only by the host), or `cudaMemAttachGlobal` (managed memory accessible by any CPUs or GPUs). Static allocation of managed memory (global scope):
```c
__device__ __managed__ int x;
```

![[Pasted image 20250601183020.png | 450]]

###### Example w/out UM
Example with the «old» programming approach with **explicit memory transfers**
```c
#include<stdio.h>
__global__ void write_value(int *ptr, int v)
{
	*ptr = v; // stub kernel run by 1 thread in 1 block only
}

int main()
{
	int *d_ptr = nullptr;
	cudaMalloc(&d_ptr, sizeof(int));
	write_value<<<1, 1>>>(d_ptr, 1);
	int host_v;
	cudaMemcpy(&host_v, d_ptr, sizeof(int), cudaMemcpyDefault);
	printf(”Value = %d\n", host_v);
	cudaFree(d_ptr);
	return 0;
}
```

Standard allocation of global memory with `cudaMalloc` (accessible by the GPU only). Explicit memory transfers via `cudaMemcpy`. We omit the direction of the transfer on GPUs supporting **UVA** (however, we have still to transfer data to allows the GPU to access it)

Example with the «new» paradigm allowed by managed **memory allocation (UM)**:
![[Pasted image 20250601183308.png]]

Memory allocated with `cudaMallocManaged`. No explicit memory transfer (transfers are done by the CUDA driver under the hood automatically). The buffer pointed by `*s` is directly accessible by the CPU and the GPU (standard allocation flag is `cudaMemAttachGlobal`)

### UM on Pre-Pascal
- **Initialization**: managed memory is set on the device memory. 
- **CPU access**: since the logical page is mapped to a physical page in GPU memory, this generates a page fault by the host. The logical page is moved to a host available memory page by copying the data from the GPU
- **GPU access**: before launching a kernel, all managed buffers are migrated «en-mass» to the device memory (even data that your kernel may not appear to explicitly touch)

![[Pasted image 20250601183734.png | 450]]

In pre-pascal we have the following issues:
- **Page faults** cannot be handled by pre-Pascal GPUs
- Pages migrate to GPU only at kernel launch – **cannot migrate on-demand**
- GPU always performs address translation during the kernel
- The host cannot access UM when a kernel is running on the device

### UM on Pascal
With CUDA 8.0 and Pascal GPUs an later models, UM has been enhanced with hardware page fault management directly by the GPU. **Consequence**: UM is not migrated via «bulk transfers» on the
GPU upon kernel launch and back upon `cudaDeviceSynchronize`. Everything happens on-demand by both the CPU and the CPU sides.

![[Pasted image 20250601184007.png | 450]]
That allows to have:
- **Page faults** can be handled by Pascal+ GPUs
- Pages migrate to GPU on-demand while the kernel is running
- GPU always performs address translation during the kernel
- The host can access UM while a kernel is running on the device (race conditions might happen)

### Memory Over-subscription
This new feature allows GPU **memory over-subscription** (i.e., we can allocate more logical managed memory than the available physical memory available on the GPU)

```c
void foo()
{
	// assume the GPU having 16 GiB memory and we allocate 64 GiB
	char *data;
	// be careful with size type
	size_t size = 64ULL*1024*1024*1024;
	cudaMallocManaged(&data, size);
}
```

In the above code, we suppose a **Pascal+ GPU** with **16 GiB** of off-chip memory. The required managed allocation is of **64 GiB** (so > 16 GiB). Pascal+ supports allocations where only a subset of pages reside on GPU. Pages can be migrated to the GPU on demand. The above allocation fails on **pre-Pascal GPUs** (e.g., Tesla, Kepler, Maxwell)

### Concurrent Accesses
Unified memory on Pascal+ GPUs allows **concurrent accesses** to unified memory locations by the CPU and the GPU simoultaneously. **Example**:
```c
__global__ void mykernel(char *data)
{
	data[1] = ‘g’;
}

void foo()
{
	char *data;
	// allocate 2 bytes of managed memory
	cudaMallocManaged(&data, 2);
	mykernel<<<1,1>>>(data);
	// no synchronization here
	data[0] = ‘c’;
	cudaDeviceSynchronize();
	cudaFree(data);
}
```

Pascal+ GPUs has the property `cudaDevAttrConcurrentManagedAccess` enabled. Pre-Pascal GPUs not. While the GPU is working `on data[0]`, the GPU is using `data[1]`. No guarantee about **ordering constraints** of memory write. This code does not work on pre-Pascal GPUs (**bus error**)

### System-wise Atomics
So far, we have studied atomic instructions executed within a kernel such as the `atomicAdd`. The default policy is `cuda::thread_scope_device` (i.e., atomic for all CUDA threads in the current program executing on the same compute device as the current thread). On Pascal+ GPUs, there is support for **system-wise atomics** whose atomicity is provided for all GPUs (through NVLINK) and the CPUs (through PCIe).

```c
__global__ void mykernel(int *addr)
{
	atomicAdd_system(addr, 10);
	// needs devices with CC 6.x
}

void foo()
{
	int *addr;
	cudaMallocManaged(&addr, sizeof(int));
	*addr = 0;
	mykernel<<<...>>>(addr);
	__sync_fetch_and_add(addr, 10);
	// CPU atomic operation
}
```

### Pre-fetching with UM
The kernel below on **Pascal+ GPUs** runs much slower than the **pre-Pascal UM** case or the **non-UM case**. Each page fault on the GPU triggers a **service overhead**. Relying on page faults to move large amounts of data, page-by-page, with overhead on each page, is inefficient. For **bulk movement**, a single “memcpy-like” operation is much more efficient.

```c
__global__ void kernel(float *data)
{
	int idx = …;
	data[idx] = …;
}

int main()
{
	…
	int n = 256*256;
	// 64K floats
	float *data;
	cudaMallocManaged(&data, n*sizeof(float);
	// 262KiB
	kernel<<<256,256>>>(data);
	…
}
```

To mitigate such a performance issue, CUDA introduced some primitives to **prefetch** data from a memory region on the host/device to another region on the device/host.

```c
cudaError_t cudaMemPrefetchAsync(const void *ptr, size_t count, 
								 int dstDevice, cudaStream_t stream = 0);
```

Such a primitve is a sort of «unified-memory alternative» to `cudaMemcpy(Async)` that complicates UM programming.

```c
__global__ void kernel(float *data)
{
	int idx = …;
	data[idx] = …;
}
int main()
{
	int n = 256*256;
	// 64K floats
	int ds = n*sizeof(float);
	float *data;
	cudaMallocManaged(&data, ds);
	// 262KiB
	cudaMemPrefetchAsync(data, ds, 0);
	// prefetch on GPU 0
	kernel<<<256,256>>>(data);
	cudaMemPrefetchAsync(data, ds, cudaCpuDeviceId);
	// prefetch on CPU
}
```

### Explicit Memory Hints
To optimize the use of UM on Pascal+ GPUs, the programmer can specify some **memory hints**. CUDA provides a specific primitive to configure memory hints (it does not trigger any data movement)

```c
cudaError_t cudaMemAdvise(const void *ptr, size_t count, 
						  cudaMemoryAdvise advice, int device);
```

We have three possible values of the `advice` parameter:

###### Advice-1 (`cudaMemAdviseSetReadMostly`)
Set an UM buffer as mostly read-only. The driver makes a «local copy» of such a buffer for each **entity** touching it. If an entity writes to it, this invalidates all copies except the one written

![[Pasted image 20250601202213.png | 500]]

###### Advice-2 (`cudaMemAdviseSetPreferredLocation`)
Set the preferred physical location where an UM buffer is stored. Therefore, if the buffer is accessed from another entity, it is not migrated but it is **directly mapped** upon page faults.

![[Pasted image 20250601202312.png | 500]]

###### Advice-3 (`cudaMemAdviseSetAccessedBy`)
Specify an entity as the one that likely accesses a buffer. That entity never generates a page fault in accessing such an UM buffer.

![[Pasted image 20250601202348.png | 500]]

# References