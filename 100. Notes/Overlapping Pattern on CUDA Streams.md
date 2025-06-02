**Data time:** 12:58 - 02-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Concurrency and Streams]]

**Area**: [[Master's degree]]
# Overlapping Pattern on CUDA Streams

Overlapping data transfers with kernel execution is often pivotal in CUDA programs. Some constraints are required to enable such an optimization:
- The device should be capable of running kernels and copies concurrently
- The kernels and the data transfers should use **different user-defined streams**
- The host memory involved in the transfers must be **[[Pageable and Pinned host memory|pinned]]**

###### Example
We have a very large array of $N$ elements and we decide to split it in partitions of $M$ elements
- **Assumption**: each element is computed independently from the others (i.e., the computation is a [[Map Parallelization|Map]])
- **Idea**: we use N/M user-defined streams, and we execute pipelines `[H2D; kernel; D2H]` each working on a different partition of the original array on a different CUDA stream
- **Consequence**: we overlap the data transfers of the partitions with the kernels working on other partitions of the array

Idea of the execution trace **without** such an **optimization**. One bulk H2D transfer, monolithic kernel, and one bulk D2H transfer:
![[Pasted image 20250602131339.png | 500]]

Below the idea with the proposed optimization. We assume N/M = 3 partitions
![[Pasted image 20250602131703.png |500]]

We still transfer one partition at a time, however, we overlap data transfers with kernel calculation on the device. One problem (similar to the **[[Unpack-Compute-Pack Pattern|optimal communication grain]]**) is to find the **best partition size**.

The host program enforcing the previous optimization pattern is the following ($M$ is the partition size in terms of elements, `bytesPerStream` is the partition size in bytes)

```c
...
for (int i = 0; i < nStreams; i++) {
	int offset = i * bytesPerStream;
	cudaMemcpyAsync(&d_A[offset], &A[offset], bytePerStream,
					cudaMemcpyHostToDevice, streams[i]);
	kernel<<<M/blockSize, blockSize, 0, stream[i]>>>(d_A, offset);
	cudaMemcpyAsync(&A[offset], &d_A[offset], bytesPerStream,
					cudaMemcpyDeviceToHost, streams[i]);
}
for (int i = 0; i < nStreams; i++) {
	cudaStreamSynchronize(streams[i]);
}
...
```

This generares several pipelines of **H2D-Kernel-D2H** tasks, each executed on a distinct CUDA stream with overlapping (one H2D, one kernel and one D2H running in parallel)

### Streamed Vector-SUM
Example to understand the potential benefit of using CUDA streams. 
- **Problem**: summing two vectors of size N each

- **Basic approach**: the program copies in device memory the first array A, then the array B, next it launches the kernel, and eventually copies the result array C in the host memory
![[Pasted image 20250602132318.png | 400]]

- **Optimization**: partitioning of the three vectors and execution using CUDA streams
![[Pasted image 20250602132342.png | 450]]

Allocate the buffers using pinned memory to overlap copies with kernel executions
```c
int *h_A, *h_B, *h_C;
cudaMallocHost((void **) &h_A, N * sizeof(int)););
cudaMallocHost((void **) &h_B, N * sizeof(int)););
cudaMallocHost((void **) &h_C, N * sizeof(int)););
```

Allocate the buffers in device memory and creation of the CUDA streams. Each CUDA stream works on a partition of size `iElem` elements:
```c
int *d_A, *d_B, *d_C;
cudaMalloc((int **) &d_A, N * sizeof(int));
cudaMalloc((int **) &d_B, N * sizeof(int));
cudaMalloc((int **) &d_C, N * sizeof(int));
...
int iElem = N / NSTREAM;
size_t iBytes = iElem * sizeof(int);
cudaStream_t stream[NSTREAM];
for (int i=0; i < NSTREAM; ++i) {
	cudaStreamCreate(&stream[i]);
}
```
Kernel doing the processing to the array partitions
```C
__global__ void sumArrays(int *A, int *B, int *C, int size)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < size) {
		C[idx] = A[idx] + B[idx];
	}
}
```

Host program running the pipelines on GPU
```C
// initiate all work on the device asynchronously in depth-first order
for (int i=0; i < NSTREAM; i++) {
	int ioffset = i * iElem;
	cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes,
					cudaMemcpyHostToDevice, stream[i]);
	cudaMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes,
					cudaMemcpyHostToDevice, stream[i]);
	sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[ioffset],
											 &d_B[ioffset], &d_C[ioffset], iElem);
	cudaMemcpyAsync(&h_C[ioffset], &d_C[ioffset], iBytes,
					cudaMemcpyDeviceToHost, stream[i]);
}
```

#### Performance Results
We show below the results (**latency**) of the vectorSUM kernel with different number of streams

![[Pasted image 20250602132715.png | 300]]

Improvement of 13% at most. Minimum around **8 streams**. Increasing the number of streams is detrimental (overheads are not amortized)
# References