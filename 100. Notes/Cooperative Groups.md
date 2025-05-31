**Data time:** 11:55 - 31-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Execution Model]]

**Area**: [[Master's degree]]
# Cooperative Groups

We have studied that a kernel is a grid of threads, combined into blocks. Grids and blocks can be 1D, 2D or 3D. Operation: **SIMT** (clever mixing between [[SIMD (Single Instruction, Multiple Data)]] and [[MIMD (Multiple Instruction, Multiple Data)]] at the architectural level).

**Idea**: threads waiting for the result of an instruction? Use computational resources with other threads in meantime. More warps are eligible and can be scheduled on the SM cores. Group of threads execute in lockstep: **warp** (currently 32 threads)
- Same instructions 
- Branching possible
- Predicates (and maskes)

**[[Shared Memory Architectures|Shared Memory]]**: fast, on-chip shared between threads of the same block
**[[SIMT and Synchronization|Synchronization]]** between threads of the same block `__syncthreads()` barrier for all threads of the same block

The **Motivation** is that not all the algorithms map easily to the available synchronization methods. Synchronization should be more flexible. The **Cooperative Groups (CG)**, introduced with CUDA 9.0, make
groups of threads explicit entities, and provide a flexible model for synchronization and communication within groups. They can be used with a proper **include** an **namespace**
```c++
#include<cooperative_groups.h>
using namespace cooperative_groups;
namespace cg = cooperative_groups;
...
```

The use of CG allows thread synchronization in a large spectrum of use cases:
- Between threads of the same block
- Between blocks of the same grid
- Between grids running on the same device
- Between grids running on different devices of the same machine

###### Dividi-and-Impera
Start with **blocks** of a certain size. Divide into smaller **sub-groups**. Continue diving, if the algorithm makes it necessary. Methods for dynamic or static divisions (**tiles**). In each level:
- threads of group have a **unique ID** (a **local index** instead of a global one)
- Use **cooperation primitives** between threads of the **same warp** (e.g., barriers, or others)

![[Pasted image 20250531121610.png | 250]]

The figure below summarizes the different cooperation levels that are possible in CUDA using Cooperative Groups. Groups can be created **dynamically** (i.e., at runtime) or **statically** (i.e., at compile time).

![[Pasted image 20250531121658.png | 600]]

**Thread group** is the base type, whose implementation depends on its construction. This concept unifies the various group types into one general, collective, thread group. We need to extend the CUDA programming model with **handlers** that can represent the groups of threads that can communicate/synchronize with each other.

![[Pasted image 20250531121920.png | 550]]
- **Grid Group**: all the threads of the kernel
- **Multi-grid Group**: all the threads of different kernels on different GPU
- **Coalesced Group**: all the threads not [[Control Divergence on NVIDIA|divergence]] in the warps
- **Thread Block**: all the threads of the block
- **Thread Block Tile**: subset of the threads of the block

What can we do with CG?
![[Pasted image 20250531121946.png | 500]]

It is an implicit group composed of all the threads in the thread block Implements the same interface as `thread_group`

![[Pasted image 20250531123711.png | 450]]

We can get the block size, its dimensions (x, y and z), and the rank (identifier) of the calling thread in the block.

###### Example: Print Rank
Example below to show a first utilization (trivial) of GC. We create a kernel running a 1D grid of one 1D block only. The block is composed of 23 threads. Each thread prints its unique local identifier inside the block.

```c++
__device__ void printRank(cg::thread_group g) {
	printf("Rank %d\n", g.thread_rank());
}

__global__ void allPrint() {
	cg::thread_block b = cg::this_thread_block();
	printRank(b);
}

int main(int argc, char **argv) {
	cudaSetDevice(0); // set the working device
	allPrint<<<1, 23>>>(); // launch the kernel
	gpuErrchk(cudaPeekAtLastError());
	cudaDeviceSynchronize(); // wait for the kernel completion
}
```

The block is represented by the object **cg::thread_block**, which is the same for all calling threads within the same block.

### Generic Reduce
Example of a simple reduce function performed by all threads of a given group (e.g., a **block** or a **warp**)

![[Pasted image 20250531124653.png]]

In the code, g is the current group where the thread belongs to. Each thread of the group executes the function `reduce` in parallel, each with its value `val` to be reduced with the others. The second input argument `smem` is a pointer to an array where the threads compute the reduce result over all the values of val for each thread.

Code below is the kernel invoking the reduce function shown in the previous slide to compute the reduce over a large array in global memory (assume `L%BLOCK_DIM == 0`)
```c++

// This function is executed by all thread in each blocks
__device__ int reduce(cg::thread_group g, int *smem, int val)
{
	int id = g.thread_rank();
	for (int i=g.size()/2; i>0; i /=2) {

		// Each thread copy is value on smem 
		smem[id] = val;
		g.sync();
		// The first half of thread group take the value from the other half
		// and sum it on its value (each iterations we divide /2 and the half decrease)
		if (id < i) {
			val += smem[id + i];
		}
		g.sync();
	}
	return val;
}

  

__global__ void call_reduce(int *array, int *reduce_result)
{
	// this array is shared by all the array of the block, for this reason we allcate
	// it with the BLOCK_SIZE
	__shared__ int array_smem[BLOCK_SIZE];
	int myval = array[(blockIdx.x * blockDim.x) + threadIdx.x];
	array[(blockIdx.x * blockDim.x) + threadIdx.x] = 0;
	cg::thread_block g = cg::this_thread_block();
	
	#if defined (BLOCK_BASIS)

	// Each thread call the reduce but only the first of each block has the right value
	// it use an atomicAdd to sum it to the final result
	int result = reduce(g, array_smem, myval);
	if (g.thread_rank() == 0) { // g.thread_rank() == threadIdx.x
		atomicAdd(reduce_result, result);
	}
	#endif
	
	#if defined (WARP_BASIS)
	// We split furhter the thread in tiles (using warp size)
	int tileIdx = g.thread_rank() / 32; // g.thread_rank() == threadIdx.x
	int *ptr = array_smem + (tileIdx * 32);
	auto tile = cg::tiled_partition(this_thread_block(), 32);
	
	int tile_result = reduce(tile, ptr, myval);
	if (tile.thread_rank() == 0) {
		atomicAdd(reduce_result, tile_result);
	}
	#endif

}
```

The code can be executed on a per-block or per-warp basis depending on the macro provided at compile time.
### Thread Block Tile
Define a sub-group of threads belonging to the same block and warp (i.e., a tile). Supported sizes: 2, 4, 8, 16 and 32. Threads of the block are divided in **row-major order**.

![[Pasted image 20250531130452.png | 600]]

We are creating tiles with a **static size**. This enables additional member functions to work with threads of the same group (i.e.,**warp-level cooperation primitives**). **Example**: one thread of a group can read the register used by another thread of the same group!

###### Static Tile Reduce
The code below calculates the reduce over an array in global memory by computing the reduce at a tiled level, using warp-level cooperation primitives. **No need to allocate and use SMEM**, everything with registers.

```c++
template<unsigned int size>
__device__ int tile_reduce(cg::thread_block_tile<size> g, int val)
{
	for (int i=g.size()/2; i>0; i /=2) {
		// Every thread execute this istructions this function allow to read
		// the val value from the thread position i (local index + i)
		val += g.shfl_down(val, i);
	}
	return val;
}

__global__ void call_reduce(int *array, int *reduce_result)
{
	int myval = array[(blockIdx.x * blockDim.x) + threadIdx.x];
	array[(blockIdx.x * blockDim.x) + threadIdx.x] = 0;
	
	auto tile = cg::tiled_partition<TILE_SIZE>(this_thread_block());
	int tile_result = tile_reduce<TILE_SIZE>(tile, myval);
	if (tile.thread_rank() == 0) {
		atomicAdd(reduce_result, tile_result);
	}
}
```

Some C++ features (e.g., generic programming) are used in the code above). We can use this functionality only in case of use static tile `TILE_SIZE` can't be bigger than 32.
### Warp-level Cooperation
List of available warp-level cooperation primitives:

![[Pasted image 20250531132917.png | 600]]

##### Voting
```c
bool p2 = g.all(p1);
```
Return 1 to all threads in the tile when the variable `p1` in all threads is equal to 1 (zero otherwise)

![[Pasted image 20250531133027.png | 400]]

```c
bool p2 = g.any(p1);
```
Return 1 to all threads in the tile when the variable `p1` in at least one thread is equal to 1 (zero otherwise)

![[Pasted image 20250531133100.png | 400]]

```c
uint n = g.ballot(p);
```
Set bit i of integer n to value of p for thread i i.e., get bit mask as an integer

![[Pasted image 20250531133158.png|400]]
##### Match
Introduced with Volta GPUs (i.e., compute capability at least 7.0). Return the set of threads that have the same value as a bitmask

```c
uint m = g.match_any(key);
uint m = g.match_all(key, &pred);
```

![[Pasted image 20250531133237.png | 500]]

They are low-level primitives that can, however, be very useful in several practical cases
### Coalesced Groups
It is possible to discover the set of coalesced threads, i.e., a group of converged threads executing in [[SIMD (Single Instruction, Multiple Data)]]

![[Pasted image 20250531133323.png | 600]]
The figure above shows the use of coalesced groups. Depending on the branch in the code, we can identify the group of threads running the same branch in a SIMD fashion. The rank is local to a group (i.e., starting from 0 to N - 1)

Coalesced groups and warp-level cooperation primitives can be used to perform **opportunistic computing** between threads of the same warp. Example, writing an optimized `myAtomicAdd` (the same reasoning can be extended to other **atomic aggregates**)

```c++
__device__ int myAtomicAdd(int *p)
{
	cg::coalesced_group g = coalesced_threads();
	int prev;

	// This is usefull to perfome one only atomic add with p + all thread that
	// called myAtomicAdd, that is possible with coalesced_threads
	if (g.thread_rank() == 0) {
		prev = atomicAdd(p, g.size());
	}

	// We propagate the value for each thread
	prev = g.thread_rank() + g.shfl(prev, 0);
	return prev;
}
```

We reduce the number of atomic simultaneous operations on the counter p in GMEM. For each coalesced group of converged threads, only the first one performs the aggregation using the right size and one atomic instruction only
### Grid Group
A set of threads within the same grid, guaranteed to be resident on the device. They can be considered as a single group with synchronization primitives.

```c++
__global__ kernel()
{
	cg::grid_group grid = this_grid();
	…
	grid.sync();
	…
}
```

This can be run if some conditions are satisfied:
- The device needs to support the `cooperativeLaunch` property
- The kernel must be launched with API `cudaLaunchCooperativeKernel`() instead of the `<<<,>>>` syntax
- All blocks of the grid must be **co-resident** on device (use the occupancy calculator)

This can be checked with the following CUDA primitive:
```c
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, numThreads, 0);
```

### Multi-Grid Group
Group of blocks across multiple devices. A set of threads guaranteed to be resident on the same system, on multiple devices.
```c
__global__ void kernel()
{
	cg::multi_grid_group multi_grid = this_multi_grid();
	…
	multi_grid.sync();
	…
}
```

![[Pasted image 20250531133833.png]]

This can be run if some conditions are satisfied:
- Devices supporting the `cooperativeMultiDeviceLaunch` property
- All blocks of the kernels **co-resident** on the respective device
- The kernels must be launched with API `cudaLaunchCooperativeKernelMultiDevice()`
# Reference