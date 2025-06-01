**Data time:** 21:28 - 01-06-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Memory Model]]

**Area**: [[Master's degree]]
# Thread Coarsening

It is a technique that we have used in previous parts of the course (e.g., for the **[[Reduction in CUDA|reduce]]** implementation in CUDA). Let’s study it more in detail. So far, most of our kernels were configured at the **finest granularity,** i.e., **each CUDA thread mimics a VP doing the smallest work unit as possible**
- **Pros**: transparent scalability, i.e., if we have enough hardware resources we can exploit parallelism at the highest degree.
- **Cons**: if hardware resources are not enough, and threads are serialized, this might generate additional overheads (e.g., warp scheduling, register allocation, etc…)

We can mitigate it by assigning each CUDA thread to multiple units of work, which is often referred to as **thread coarsening**. This can have several benefits, non only in terms of **performance** but also related to **debugging**, **portability**, and **readability** of the code.

### Grid-Stride Loops
Let us consider a very simply kernel doing the **vector addition**. Basic implementation (hereinafter called **monolithic**)
```c
__global__ void vectAdd(int N, float a, float *x, float *y)
{
	int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < N)
		y[i] = a * x[i] + y[i];
}
```

The version above assumes a single large 1D grid of threads to process the entire arrays in one pass. It is often better to use a **grid-stride for loop** as below
```c
__global__ void vectadd(int N, float a, float *x, float *y)
{
	int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	for (int i = ix; i < N; i += (blockDim.x * gridDim.x)) {
		y[i] = a * x[i] + y[i];
	}
}
```
We no longer assume that the thread grid is large enough to cover the entire data array (i.e., it is generally smaller).

The stride of the loop is `blockDim.x * gridDim.x`, which is the total number of threads in the **1D grid** (we use **1D blocks**). So, if there are **1280** threads in the grid, **thread 0** will compute elements **0, 1280, 2560**, etc.

![[Pasted image 20250601214032.png | 450]]

No overhead if launched with a grid large enough to cover all iterations of the loop (same case of the monolithic kernel). There are several advantages of this simple optimization (it is more a **design pattern** for CUDA kernels)
###### Scalability and thread reuse
We can support **any problem size** even if it exceeds the largest grid size your CUDA device supports. Moreover, you can limit the number of blocks you use to **tune performance**. For example, it’s often useful to launch a number of blocks that is a multiple of the number of multiprocessors on the device, to balance utilization.
```c
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
// Perform vector addition on 1M elements
vectadd<<<32*numSMs, 256>>>(1 << 20, 2.0, x, y);
```

Thread reuse amortizes **thread creation** and **destruction cost** along with any other processing the kernel might do before or after the loop (such as **thread-private** or **shared data initialization**).
###### Debugging
We can easily switch to serial processing by launching one block with one thread only
```c
vectadd<<<1,1>>>(1<<20, 2.0, x, y);
```
# References