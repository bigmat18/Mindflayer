**Data time:** 15:41 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Architectures and Compiler]]

**Area**: [[Master's degree]]
# Inspecting the GPU Properties

Sometimes, we need to query the device properties using the **CUDA API** and/or through some **command-line utilities**. Queries: how many devices; how many SM; how many resident threads per SM; how much memory; etc...

```c
int dev_count;
cudaError_t cudaGetDeviceCount(&dev_count); // get number of GPUs
```

```c
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
```

The second call returns a large set of device information in a struct named cudaDeviceProp (link). Example of fields that can be extacted:
- Shared memory size available per block
- Number of registers per SM
- Number of SMs
- Max number of threads per block
- Max number of resident blocks per SM

**Example of utilization of the CUDA API**
We want to choose (and so to launch kernels) on the GPU available in our system having the highest number of SMs.
```c
int numDevices = 0;
cudaGetDeviceCount(&numDevices);
if (numDevices > 1) {
	int maxMultiprocessors = 0, maxDevice = 0;
	for (int device=0; device<numDevices; device++) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, device);
		if (maxMultiprocessors < props.multiProcessorCount) {
			maxMultiprocessors = props.multiProcessorCount;
			maxDevice = device;
		}
	}
	cudaSetDevice(maxDevice);
}
…
```

We get the number of devices, we iterate across them and we get the device properties. We identify the identifier of the device with the maximum number of SMs, and we set it to be used for the rest of our program.

### Nvidia-smi
It is possible to monitor the actual utilization of a GPU through the `nvidia-smi` command-line utility.

![[Pasted image 20250601154747.png|550]]

More complete report (MEMORY, UTILIZATION, ECC, TEMPERATURE, POWER, CLOCK, COMPUTE, PIDS, PERFORMANCE, SUPPORTED_CLOCKS, PAGE_RETIREMENT, ACCOUNTING). Example:

```
$ nvidia-smi -q -i 0 -d MEMORY
```


### Occupancy
An important metric to understand the effective utilization of the GPU is the so-called **occupancy** mesured as:
$$
occupancy = \frac{resident\_warps}{max\_resident\_warps}
$$
Where the **numerator** is limited by aspects related to the kernel such as the amount of used registers per thread, the amount of shared memory used by a block, and so forth. The **denominator** is limited by aspects related to the GPU device such as the memory bandwidth, the number of slots of the warp schedulers, etc...

Occupancy measures how well concurrency/parallelism provided by the device is actually utilized by our kernel. Therefore, it is important to design kernels in such a way that occupancy is maximized through a clever utilization of the resources provided by Stream Multi-processors (notably, **registers** per thread, **[[Shared Memory (SMEM) on GPU|SMEM]]** per block)

###### Example of HW Limits
- **Example 1**: suppose a [[Volta Architecture (2017)|Volta GPU (V100)]]
	- Resident warps/SM is 64
	- Resident blocks/SM is 32
	- Resident threads/SM is 64 ∙ 32 = 2048
	- Kernel with 512 threads/block → max 4 blocks/SM
	- Kernel with 128 threads/block → max 16 blocks/SM
	- Suppose that a kernel is run with 32 threads/block → since we can have at most 32 blocks/SM, we run 1024 threads/SM, so occupancy is 50%
- **Example 2**: suppose a [[Volta Architecture (2017)|Volta GPU (V100)]]
	- Registers/thread is 255
	- Registers/block is 64K (max no. of registers per SM)
	- Assume a kernel with 32 registers/thread and block size of 256 threads → 8K registers/block, 8 blocks/SM, threads/SM are 2048 (occupancy 100%)
	- Assume a kernel with 64 registers/thread, block size of 256 threads → 16K registers/block, 4 blocks/SM, threads per SM are 1024 (occupancy 50%)

Some CUDA API functions assist programmers in choosing the **block size** and the **grid size**. Useful CUDA primitive:
```c
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor (int *numBlocks, 
	T func, int blockSize, size_t dynamicSMemSize);
```
This function reports occupancy in terms of the number of resident thread blocks per multiprocessor (`numBlocks`). Multiplying by the number of warps/block yields the number of resident warps/SM; dividing it by max warps/SM gives the occupancy as a percentage. Another useful primitive:
```c
cudaError_t cudaOccupancyMaxPotentialBlockSize(int *minGridSize, 
	int *blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0)
```
It heuristically calculates an execution configuration that achieves the maximum multiprocessor-level occupancy

**Example** of the API usage:
```c
// Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	d[idx] = a[idx] * b[idx];
}


// Host code
int main() {
	int numBlocks;
	int device;
	cudaDeviceProp prop;
	int residentWarps;
	int maxWarps;
	
	
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);
	// max number of resident warp on a SM
	maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
	  
	for (int blockSize = 2; blockSize <= 1024; blockSize*=2) {

		// This function calculate how much block can be active simultaneously on a SM,
		// considering register, shared mamory and hw blocks limits
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&numBlocks, // OUTPUT: max number of active block per SM
			MyKernel, // kernel to used
			blockSize, // thread per block that we want
			0 // dynamic shared memory
		);
		residentWarps = numBlocks * blockSize / prop.warpSize;
		double occup = (double) residentWarps / maxWarps * 100;
			printf("blockSize = %4d <-> Occupancy [numBlocks = %2d, activeWarps = %2d]:\t%2.2f%%\n", blockSize, numBlocks, residentWarps, occup);
	}
	
	return 0;
}
```

### Register Spilling
If registers are not sufficient, the compiler will do **register spilling** (copying the content of registers in global memory). It is possible to limit the amount of registers used by a thread by providing the compiler option `–maxrregcount`. It is possible to inspect the amount of registers used by a thread with a compiler flag `--ptxas-options=-v`

![[Pasted image 20250601160400.png | 550]]
# References