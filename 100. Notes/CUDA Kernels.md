**Data time:** 13:07 - 28-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to CUDA]]

**Area**: [[Master's degree]]
# CUDA Kernels

A CUDA kernel is a program describing the activity performed by a potentially huge set of threads. Threads are organized in a two-level hierarchy:
- **Grid**: an ordered set of blocks
- **Blocks**: an ordered collection of threads (max 1024)

Grids and blocks can be defined in one, two, or three dimensions, yielding 9 possible combinations (although it is common that they have the same number of dimensions). 
###### Example with 1D grid of 1D blocks
![[Pasted image 20250528130953.png | 400]]

One 1D grid composed of $N>0$ 1D blocks. Each block has $M>0$ threads. The values of N and M depend on the problem size.

###### Example with 1D grid of 2D blocks
![[Pasted image 20250528131149.png]]

A grid of N blocks organized in one dimension. Each 2D block is of $\sqrt{M} \times \sqrt{M} = M$ threads
###### Example with 2D grid of 2D blocks
![[Pasted image 20250528131201.png]]
A grid of $\sqrt{N} \times \sqrt{N} = N$ blocks organized in two dimensions. Each 2D block is of $\sqrt{M} \times \sqrt{M} = M$ threads. Maximum number of blocks is $2^{32} - 1$ in the x dimension, $2^{16 -1}$ in the y and z dimensions.

### CUDA Blocks and Threads
A block is an essential logical component of a CUDA kernel. It contains several threads (max 1024). Threads in the same block are closely coupled because:
- They can be synchronized with **block-local synchronization primitives**
- They can access a **block-local shared memory** on the device, much faster than the Global Memory

Threads of different blocks (but of the same grid) can still cooperate using **Cooperative Groups**. All threads of the same kernel share the same space in Global Memory (e.g., input and output buffers). Each thread/block is uniquely identified within the block/grid through two **built-in variables**:
- The index of the block in the grid is refereed to as `blockIdx`
- The index of the thread in the block is referred to as `threadIdx`

Such indexes (`uint3`) are expressed as coordinates in each dimension. For example., `blockIdx.{x,y,z}` and `threadIdx.{x,y,z}`.

The size of a grid and a block are specified by other two built-in variables pre-initialized by the CUDA driver before calling the kernel. They are:
- **blockDim** is the size of the block in terms of the number of threads. 
- **gridDim** is the size of the grid in terms of the number of blocks. 

![[Pasted image 20250528190405.png]]

Such variables are of type `dim3`, a special vector of integers similar to the `uint3` data type (the latter specifies indexes, the former the size of each dimension). When we define a variable of type `dim3`, each unspecified component is initialized (implicitly) to 1. Each component of a variable `dim3` is accessed using the fields `x`, `y`, and `z`.

Very often, we need to identify a block in the grid with a **unique block identifier (blockUID)**. Such a linear identifier is unique in the whole grid.

![[Pasted image 20250528190439.png]]

**blockUID** is computed in one of the three following ways depending on the grid dimensionality. In the 2D case, `gridDim.x` is 8, and `gridDim.y` is 4. The red block has `blockIdx.x` equal to 5, and `blockIdx.y` equal to 2, which corresponds to the **blockUID** equal to 21.

Very often, we need to identify a thread in the grid with a **unique thread identifier (threadUID**) unique in the grid. We first have to identify the thread in its block (1D, 2D or 3D). Then, we identify the thread in the grid by calculating the blockUID as in the previous slide. Example (2D blocks)
```
threadUID = (blockUID * blockSize) + threadIdx.x + (threadIdx.y * blockDim.x)
```
Where `blockSize` is the number of threads of a block
```
blockSize = blockDim.x * blockDim.y * blockDim.z
```

### Launching CUDA Kernels
CUDA extends the C/C++ syntax. One example of this extension is related to the launching of a CUDA kernel. For example:
```c++
kernel_name <<<gridDim, blockDim>>> (argument list);
```
Where `gridDim` and `blockDim` have the `dim3` data type

![[Pasted image 20250528193148.png | 500]]

A basic "Hello world" code is:
```c
__global__ void helloWorld_GPU(void)
{
int threadUID = threadIdx.x;
printf(“Hello world from the GPU by thread %d\n”, threadUID);

```

The CUDA `__global__` decorator tells the compiler that the function `helloWorld_GPU` is a kernel and will be executed by the device. Kernels always return void, cannot use static variables, and do not support pointers to functions. Kernels accept any arbitrary sequence of input parameters. 

This kernel is a very simplified one, without input parameters. In case an **input parameter** is provided, this is copied from the host memory to the GPU memory before calling the kernel (automatically by the CUDA driver)
```c
int main(int argc, char **argv)
{
	int n = atoi(argv[1]); // n is the number of threads of the kernel
	cudaSetDevice(0); // set the working device (optional)
	helloWorld_GPU<<<1, n>>>(); // launch the kernel
	cudaDeviceSynchronize(); // wait for the kernel completion
}
```

Kernel invocation with 1 block and 𝑛 > 0 threads each one printing the string «hello world» and their unique identifier. The grid configuration is passed through `<<<…, …>>>`, where the first parameter is the number of blocks while the second is the number of threads. 

**Grid configuration** parameters are of type dim3. Here we pass integers because the grid is 1-dimensional (with 1D blocks). Kernel execution is always **asynchronous** with the host. First two CUDA API primitives: `cudaSetDevice` and `cudaDeviceSynchronize`

Kernels and functions can be decorated with proper **CUDA directives** to tell the compiler if they are kernels, and if  functions can be called by the host, by the device or by both.

![[Pasted image 20250528193858.png]]

So, we can create a portion of code that can be arbitrarily called by the device or the host based on runtime decisions. This can be useful in several practical cases
# References