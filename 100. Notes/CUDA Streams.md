**Data time:** 11:15 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Concurrency and Streams]]

**Area**: [[Master's degree]]
# CUDA Streams

A **CUDA stream** is a sequence of possibly aynchronous CUDA calls (hereinafter called **tasks**) that are executed by the device in the ordering dictacted by the host program. A stream encapsulates information about such tasks, it preserves their **FIFO** ordering, and can inspect their **status**.

If task $C_i$ proceeds task $C_j$ and they belong to the same stream, the device executes $C_j$ only when the execution of $C_j$ is complete.

![[Pasted image 20250602111909.png | 250]]

If two tasks belong to **different streams**, they can be completed in **any order** by the device (i.e., they can run in parallel, or interleaved depending on the available GPU resources). Complex GPU programs make
largely use of CUDA streams to exploit concurrency and real parallelism whenever it is possible.

**GPU tasks** that can be enqueued in a CUDA stream are **kernel launches**, **memory allocations**, **memory copies** and others. If not specified, the used stream is the default one (different semantics in older and newer GPU models regarding the use of the **default stream**).

All CUDA primitives are related to a CUDA stream also when this is not explicitly indicated. Indeed, two kinds of CUDA streams exist:
- **user-defined CUDA streams** (explicitly created by the developer using the `cudaStreamCreate` API call)
- **default CUDA stream** (also called the **null stream**)

### User Defined CUDA Streams
CUDA streams can be **created** using a specific CUDA primitive for this purpose
```c
cudaError_t cudaStreamCreate(cudaStream_t *pStream)
```

Every time we launch a CUDA kernel, we can specify a CUDA stream where that kernel will belong to
```c
kernel_name<<<grid, block, sharedMemSize, pStream>>>(arguments list);
```
A CUDA stream should always be **destroyed** before the end of the program
```c
cudaError_t cudaStreamDestroy(cudaStream_t *pStream);
```
If a stream is destroyed before the pending CUDA tasks on that stream are completed, the command returns the control to the host program only when all such tasks are complete.

Memory copies can also be issued on a given CUDA stream. **Example**, we can allocate pinned memory using one of the two primitives below
```c
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
```

Then, we perform an **asynchronous H2D data transfer** issued on a specific (previously created) CUDA stream.
```c
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
```
We can issue more copies on the same stream: they are asynchronous w.r.t the host program and will be executed and completed by the device in the **issuing order**

This idea can be exploited to **overlap** data transfers with kernel executions when we use more CUDA streams.

### Default CUDA Streams
The default stream is adopted if a user-defined stream is not specifically used by the programmer. The semantics of the default stream and its interference with the other user-defined streams is important. There are two different semantics:
- **Blocking semantics**: the execution of any task running on the default stream (i.e., memory copies, kernels) blocks the tasks running on other user-defined streams
- **Non-blocking semantics**: the default stream is treated as a user-defined stream. Full concurrency and overlapping between the default stream and the others is possible
The behavior depends on the GPU and some compilation flags.

###### Example 1: host-device perspective
```c
cudaMemcpy(d_array, h_array, numBytes, cudaMemcpyHostToDevice);
GPU_kernel<<<blocks, threads>>>(d_array);
cudaMemcpy(h_array, d_array, numBytes, cudaMemcpyDeviceToHost);
```
Considerations:
- **GPU**: H2D, kernel, D2H are executed on the **default stream** and they will be run in that order by the device
- **CPU**: H2D is **synchronous** for the host program. Furthermore, the next statement is a kernel launch that will start when the H2D copy is complete. The D2H is **synchronous** (the host program continues after the D2H when the copy is complete)

###### Example 2: simple host-device overlapping
```c
cudaMemcpy(d_array, h_array, numBytes, cudaMemcpyHostToDevice);
GPU_kernel<<<blocks, threads>>>(d_array);
function_A (...);
cudaMemcpy(h_array, d_array, numBytes, cudaMemcpyDeviceToHost);
`````
This fragment looks identical to the one above. However, the CPU executes `function_A` in parallel to the kernel.

![[Pasted image 20250602113457.png | 500]]

Function_A execution is **overlapped** with the kernel execution. The **D2H copy** is executed by the host when `Function_A` is complete, and it is further executed by the device when the kernel is complete.

#### Legacy GPU
In old GPUs with `CUDA < 7.0`, the default stream was a **special stream** (one per device) that by default synchronizes all host threads of any host program. Example:
```c
cudaStream_t streams[8];
float *data[8];
for (int i=0; i<8; i++) {
	cudaStreamCreate(&streams[i]);
	cudaMalloc(&data[i], N * sizeof(float));
}
for (int i=0; i<8; i++) {
	// launch a tiny kernel on the default stream
	k<<<1, 1>>>();
	// launch one worker kernel per stream
	kernel<<<1, 64, 0, streams[i]>>>(data[i], N);
}
cudaDeviceSynchronize();
```

On modern GPU, this legacy behavior can still be reproduced by compiling with the flag `--default-stream legacy`. The default semantics was the **blocking one**, so a task on the default stream blocks any tasks on user-defined streams. Tracing shown below:

![[Pasted image 20250602114021.png]]

Everytime we launch a GPU task (e.g., a kernel) on the default stream, we have an **implicit synchronization** on the device. All **previously launched tasks** on the device (on any CUDA stream) should complete before the task launched on the default stream can start. Furthermore, **any future task** issued on a user-defined stream can start only when the task on the default stream is complete.

#### Semantics
We understood that, in `CUDA < 7.0`, all user-defined streams are by default **blocking** w.r.t tasks triggered on the default stream. Example:
```c
kernel_1<<<1, 1, 0, stream_1>>>();
kernel_2<<<1, 1>>>();
kernel_3<<<1, 1, 0, stream_2>>>();
```
- **Kernel 2** does not start until **kernel 1** is complete 
- **Kernel 3** does not start until **kernel 2** is complete

CUDA provides the user with the possibility to change the semantics of user-defined streams w.r.t the default stream. This can be done with the following primitive
```c
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
```
Different options for the flags parameter:
- `cudaStreamDefault`: default stream semantics (blocking with the default stream)
- `cudaStreamNonBlocking`: non-blocking semantics with the default stream

###### Case 1: Default Stream
```c
kernel1<<<blocks, threads>>>();
kernel2<<<blocks, threads>>>();
```

![[Pasted image 20250602115052.png | 300]]

###### Case 2: Blocking interaction between a user-defined stream `stream1` and the default stream
```c
cudaStream_t stream1;
cudaStreamCreate(&stream1);
kernel1<<<blocks, threads>>>();
kernel2<<<blocks, threads, 0, stream1>>>();
cudaStreamDestroy(stream1);
```

![[Pasted image 20250602115141.png | 300]]

###### Case 3: Non blocking semantics
```c
cudaStream_t stream1;
cudaStreamCreateWithFlags(&stream1,
cudaStreamNonBlocking);
kernel1<<<blocks, threads>>>();
kernel2<<<blocks, threads, 0, stream1>>>();
cudaStreamDestroy(stream1)
```

![[Pasted image 20250602115221.png | 250]]

#### Per-Thread Semantics
Starting from `CUDA 7`, the default stream is treated as a user-defined stream (so with non-blocking semantics). So, all primitives called on that stream proceed independently with the calls on other user-defined streams. In the previous example, the tracing will be the following.

![[Pasted image 20250602115350.png | 300]]
- Same code, completely different performance trace 
- Kernels ”K” are executed sequentially without overlapping since they are all issued on the default stream
- However, during their execution, the GPU is capable of running other kernels issued on the other 8 user-defined CUDA streams
- This semantics can be enabled by compiling with the flag `--default-stream per-thread with CUDA >= 7.0`

### Real Example
Example of a CUDA program using multiple user-defined streams and running a simple kernel
```c
_global__ void kernel(float *x, int n)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
		x[i] = sqrt(pow(2,i));
	}
}
```

The kernel above uses the studied pattern [[Thread Coarsening|grid-stride loop]] (i.e., the kernel works for any size of the input array).

```c
...
cudaStream_t streams[num_streams];
float *data[num_streams];
for (int i = 0; i < num_streams; i++) {
	cudaStreamCreate(&streams[i]);
	cudaMalloc(&data[i], N * sizeof(float));
	kernel<<<1, 64, 0, streams[i]>>>(data[i], N);
	kernel<<<1, 1>>>(0, 0);
	// launched on the default CUDA stream
}
```

Let us use **[[CUDA Compiler and Profiling Tools|NVIDIA Nsight System]]** to profile the execution of the host program. Example:

![[Pasted image 20250602120804.png | 600]]

The tool generates a log report file (`report1.nsys-rep`) that can be inspected with the **Nsight GUI tool**. The analysis can be done on a different machine.

- Analysis of the previous code (**Version 1**):
![[Pasted image 20250602120941.png]]

- Analysis of **Version 2** of the code (where the launching statement of the kernel issued on the default stream has been commented)
![[Pasted image 20250602121017.png]]

- **Version 3** compiled with the option `--default-stream per-thread`. Default stream is a normal user-defined stream. Full concurrency is enabled: 
![[Pasted image 20250602121735.png]]

The default stream is mapped onto one of the stream above. Full overlapping, the default stream (one per host thread) does not block any task running on the other 10 user-defined streams of our program. Check the log files in the folder and try to visualize them using the Nsight Profiler tool by NVIDIA

### Implicit Synchronization
Although calls on different CUDA streams usually proceed independently (and potentially overlapped by the GPU hardware), some exceptions might happen. This is the case of **implicit synchronization conditions** in CUDA programs. 

Two tasks issued on different streams cannot run concurrently if any one of the following operations is issued in-between them by the host thread:
- **a page-locked host memory allocation**
- **a device memory allocation**
- **a device memory set**
- **a memory copy between two addresses to the same device memory** (not-stream oriented)
- any CUDA command issued to the **default stream** (with **legacy blocking semantics**)

#### Memory Allocation Issues
We know that CUDA memory allocations/deallocations generate an **implicit barrier**. Example:
```c
cudaMalloc(&ptrA, sizeA);
kernel1<<<..., stream1>>>(ptrA);
cudaMalloc(&ptrB, sizeB);
kernel2<<<..., stream2>>>(ptrB);
cudaFree(ptrA);
cudaFree(ptrB);
```
In the code above, **kernel2** is not overlapped with **kernel1** even if they are issued on two user-defined streams. The reason is that the `cudaMalloc` between them is an implicit synchronization. Possible easy solution for this use case.
```c
cudaMalloc(&ptrA, sizeA
cudaMalloc(&ptrB, sizeB);
kernel1<<<..., stream1>>>(ptrA);
kernel2<<<..., stream2>>>(ptrB);
cudaFree(ptrA);
cudaFree(ptrB);
```

`CUDA 11.2` introduced a s**tream-ordered memory allocator** to solve these types of problems. With this new CUDA feature, it is possible to bind a memory allocation/deallocation task with a CUDA stream and to execute it asynchronously w.r.t the host program. New API for allocation and deallocation
```c
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream)
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream)
```
Memory allocations are shift from global-scope operations that synchronize the entire device to stream-ordered operations. This eliminates the need for synchronizing outstanding GPU work and helps restricting the lifetime of the allocation to the GPU tasks that access it.

We can use stream-oriented allocation/deallocation primitives to avoid implicit synchronization. Example
```c
cudaMallocAsync(&ptrA, sizeA, stream1);
kernel1<<<..., stream1>>>(ptrA);
cudaFreeAsync(ptrA, stream1);
// No synchronization necessary
cudaMallocAsync(&ptrB, sizeB, stream2);
// can reuse the memory freed previously
kernel2<<<..., stream2>>>(ptrB);
cudaFreeAsync(ptrB, stream2);
```
The memory returned from `cudaMallocAsync` can be accessed by any kernel or copy operation as long as the calls are ordered to execute after the allocation operation, and before the deallocation operation, in stream order. Allocation and deallocation with stream-based semantics become like kernel launches from the ordering perspective.

### Overlapping Capabilities
Modern GPUs allow for different overlapping capabilities:
###### Overlap data transfers with kernel execution
E.g., an H2D copy on **stream1** overlapped with a kernel on **stream2** if the copy uses a pinned memory and it is asynchronous.
###### Concurrent kernels execution
Enabled from `CUDA 2.x`, modern GPUS are able to run up to 128 concurrent kernels (this can be inspected by reading the `concurrentKernels` device property)
###### Concurrent data transfers
GPUs having two DMAs can perform a H2D and a D2H data transfers in parallel, thus saturating the PCIe bandwidth (hopefully) since it is full duplex.

How can we inspect at runtime the properties regarding (for example) concurrent data transfers of our GPU?
```c
cudaDeviceProp dProp;
cudaGetDeviceProperties(&dProp, 0);
printf("Device %s asyncEngineCount %d\n", dProp.
asyncEngineCount);
```
`asyncEngineCount` is **1** when the device can concurrently copy memory between host and device while executing a kernel. It is **2** when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time. It is **0** if neither of these is supported.

![[Pasted image 20250602123334.png | 150]]

### Explicit Synchronizing 
All primitives invoked on a given CUDA stream are **asynchronous** w.r.t the host program. Sometimes, we might need to wait for the completion of all the previously executed asynchronous calls on a given stream. This can be done with the following primitive
```c
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
```
It is worth noting that the primitive `cudaDeviceSynchronize` blocks the host program until all the asynchronous calls invoked in any stream by that program are complete.

It is also possible to check whether previous calls on a given stream are complete or not (**non-blocking primitive**). This can be done with the following primitive.
```c
cudaError_t cudaStreamQuery(cudaStream_t stream);
```
The primitive returns `cudaSuccess` if all primitives on the stream are complete, `cudaErrorNotReady` otherwise.

### Cuda Streams and Priorities
On devices with **Compute Capability 3.5**, kernels can be assigned to a **priority**. If a kernel is assigned to a higher priority w.r.t a running kernel, this one can release resources to schedule the former kernel.

**Priorities have effect only on kernels** (not on data transfers). Priorities can be assigned at the **stream level** with the following API. Create a stream with a given priority:
```c
cudaError_t cudaStreamCreateWithPriority(
	cudaStream_t *pStream, unsigned int flags, int priority
);
```
To get the minimum and maximum priorities supported by the device, we can use the following primitive:
```c
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
```
If the chosen priority is out-of—range, it is automatically converted to fall in the given range.

### Host Callbacks
It is possible to ask the CUDA driver to run a given **host function** at a certain point of a CUDA stream. Such functions are called **CUDA host callback**. 
```c
void CUDART_CB MyCallback(void *data)
{
	printf("Inside callback %d\n", (size_t) *data);
}

for (size_t i=0; i<2; i++) {
	cudaMemcpyAsync(d_in[i], h_in[i], size, cudaMemcpyHostToDevice, stream[i]);
	MyKernel<<<100, 512, 0, stream[i]>>>(d_out[i], d_in[i], size);
	cudaMemcpyAsync(h_out[i], d_out[i], size, cudaMemcpyDeviceToHost, stream[i]);
	cudaLaunchHostFunc(stream[i], MyCallback, (void *) &i);
}
```
The CUDA host callback is executed by a **thread** of the **CUDA driver** (not one of our host program !). The callback is executed respecting the **stream order**: in the code above after the H2D, kernel, D2H pipeline of the same stream (but in parallel with any task of other streams). Useful to register a work to do upon kernel completion.

### [[HYPER-Q]]

### [[Overlapping Pattern on CUDA Streams]]

# References