**Data time:** 10:21 - 31-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Execution Model]]

**Area**: [[Master's degree]]
# SIMT and Synchronization

With the **[[Control Divergence on NVIDIA|legacy thread scheduling (i.e., pre-Volta)]]**, threads on divergent branches cannot execute synchronization primitives. What would happen if threads in the same warp synchronize with each other?
###### Example

![[Pasted image 20250531102238.png | 550]]

On pre-Volta GPUs, the kernel hangs since threads cannot execute a divergent branch until the first chosen branch is complete. So, on divergent branches no synchronization is possible.

###### Example
**[[Event Notification]]** with shared boolean flags. The synchronization performed between threads on the same warp is a **busy-waiting spin-loop** on a Boolean **flag** allocated in global memory
```c
__global__ void prova_kernel(volatile bool *readyA, volatile bool *readyB)
{
	if (threadIdx.x == 0) {
		; // A;
		*readyA = true;
		while(!(*readyB));
	}else if (threadIdx.x == 1) {
		; // B;
		*readyB = true;
		while(!(*readyA));

	}
	; // C;
}
```

if we compile with `-gencode arch=compute_80,code=sm_80` (support to a recent Ampere GPUs), the program works well. Statements of each divergent block are executed in an interleaved manner. If we compile with `-gencode arch=compute_60,code=sm_80` (support to old pre-Volta GPUs only) the program hangs forever.

On post-Volta GPUs, the kernel above completes correctlybecause threads have their PCs, and instructions of divergent branches are executed in an interleaved manner

### Block-level Synchronization
It is often useful to provide some kinds of synchronization between CUDA threads. **Block-level synchronization** can be used among threads belonging to the **same block**. Only when all threads of the same block reach the **[[Barriers]]**, they can resume their execution.

![[Pasted image 20250531113649.png| 400]]

Block-level synchronization can be used in a kernel code with the intrinsics `__syncthreads()`. It takes no parameters and returns void.
### System-level Synchronization
Block-level synchronization should be not confused with **system-level synchronization**. The primitive `cudaDeviceSynchronize` can be used to wait for the completion of all preceding pending tasks running on the device (e.g., kernels, H2D , and D2H memory copies). It blocks the host program until this condition is satisfied.
```c
inline uint64_t current_time_nsecs()
{
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return (t.tv_sec)*1000000000L + t.tv_nsec;
}
int main()
{
	…
	uint64_t initial_time = current_time_nsecs();
	…
	kernel_name<<<…>>>(…);
	…
	cudaDeviceSynchronize();
	uint64_t end_time = current_time_nsecs();
	uint64_t elapsed = end_time - initial_time;
	…
}
```

- Without the `cudaDeviceSynchronize` the kernel on the device goes in parallel with the host code
- The `cudaDeviceSynchronize` returns an error if any preceding task on the device failed

### Warp-level Synchronization
It is possible to synchronize threads of the same kernel in a more fine-grained manner. For example, synchronizing threads of the **same warp** that executed divergent branches to reconverge. This can be done with the `__syncwarp(mask)` intrinsics.

![[Pasted image 20250531114207.png | 550]]

### Deadlocks
Both in **pre-** and **post-Volta GPUs**, **deadlocks** might occur if we are not careful in implementing our kernels with thread synchronization. [[Barriers]] can be executed in divergent branches (in **post-Volta**
**models**), however we must be careful that the barrier will be executed by all threads in all divergent branches.
```c
__global__ void kernel_test(…)
{
	unsigned int tUID = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (condition 1) {
		A;
		__syncthreads();
	} else {
		B;
	}
}
```

If **all** threads enter the if branch (or all enter the else branch) no issue arises and the kernel completes. Otherwise, some threads will hang forever and the kernel never terminates (both in pre- and post-Volta GPUs).


# References