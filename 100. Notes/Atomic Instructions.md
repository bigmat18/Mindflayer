**Data time:** 11:49 - 31-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[CUDA Execution Model]]

**Area**: [[Master's degree]]
# Atomic Instructions

NVIDIA GPU ISA provides a set of **atomic RMW instructions**, like CPU ISAs. They can be used to provide synchronization during accesses to shared data in memory by threads of the same kernel.

**Example**: The function `int atomicAdd(int *p, int v)` adds the value v to the integer pointed by p in an atomic manner.

Other instructions of this kind exist and we will sometimes introduce them in the course when needed and useful. However, be careful (poor performance).
```c
__global__ void kernel_test(int *A)
{
	atomicAdd(A, 1);
}
```

The additions performed by all threads serialize (they are executed in a sequential fashion). **No speedup**.
# References