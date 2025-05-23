**Data time:** 13:37 - 22-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[SIMD and GPU]]

**Area**: [[Master's degree]]
# Vectorization Instructions

The basics idea is to provide a new set of machines instructions working on **arrays** instead of **scalar registers**. An example below:

```c
int A[L], B[L], C[L]; // three arrays
... // initialization of A and B
for i = 0 to L – 1 do
	C[i] = A[i] + B[i];
```

```
LOOP: LOAD RA-base, Ri LSL #2, Ra
	  LOAD RB-base, Ri LSL #2, Rb
	  ADD Ra, Rb, Rc
	  STORE RC-base, Ri LSL #2, Rc
	  INCR Ri
	  IF Ri<RL LOOP
```

LOAD and STORE multiply the index of the loop in `Ri` by 4 to correctly access the arrays (memory addressable at the byte). A SIMD program is instead a sequential ordering of **SIMD instructions**, each expressing a specific function over arrays

**Example**: `ADDV RVA, RVB, RVC`
RVA, RVB, RVC are vector registers containing $n > 1$ elements of 32 or 64 bits (for the sake of homogeneity with the previous part of the course, we suppose 32-bit words from now on). Possible values of 𝒏 are 2, 4, 8 operands per register

### SIMD Instructions on Pipelined PE
Suppose to execute the previous code on a **scalar [[Pipeline Processors]]** with the following structure (no optimizations)

![[Pasted image 20250522135745.png | 600]]

Each iterations consts 16 clock cycles to execute six instructions. Several wasted clock cycles due to data and control dependencies (**CPI penalty** is of 10/6 = 1.66).

![[Pasted image 20250522135903.png | 600]]

SIMD instructions extend the instruction set of traditional [[SISD (Single Instruction, Single Data)]]/[[MIMD (Multiple Instruction, Multiple Data)]] machines to provide operations on arrays.

In the previous example assume $n = 4$ and $L \mod 4 = 0$ (the whole arrays can perfectly be partitioned into 4-elements partitions)

![[Pasted image 20250522140159.png | 400]]

- SIMD extensions allow **ammortizing** the **cost** of **Fetch** and **Decode** phases
- SIMD extensions also allow fore reducing the frequency of branches

The structure is a loop of $L/n$ iterations. Each LOADV loads $n=4$ cosecutive 32-bit words from the memory starting from the logical address obtained as `RG[RA/B-base] + RG[Ri] * 4`
The vector is stored in a register with width **n** distinct 32 bits words. In this case $n=4$ each vector register is of **128 bits**

Pipelined PE equipped with a more powerful Execution Unit (called **Vectorized Execution Unit**). VEU incorporates a powerful **ALU** to work on wider registers. The decode stage has a **scalar RF** and a **vector RF**

![[Pasted image 20250522141112.png | 600]]

### Floating-Point Instructions

PEs are equipped with **FP registers** and an ALU specialized with operations on FPs (**IEEE 754** single and double precision). **Scalar FP instructions** at least. Example:
```c
float A[L], C[L]; // two arrays
... // initialization of A
for i = 0 to L – 1 do
	C[i] = sqrt(A[i]);
```
Compilation in D-RISC, scalar version (left), **[[Vectorization Instructions]]** (right)

![[Pasted image 20250522142003.png]]

Same structure but a different number of iterations since each vector instruction computes four positions of the output array. The wider the vector registers are, the greater the benefit.

Now we see a bit more complex example:

```c
float A[L], B[L], C[L]; // three arrays
... // initialization of A and B
for i = 0 to L – 1 do
	C[i] = sqrt(max(A[i], B[i]));
```

Assume instruction on $n = 4$ 32-bit words (vector registers of **128 bits**). The problem is how to implement conditions. Use a **vector flag register** `Rvmask` of four bits. Use a vectorized **compare** instruction to set the flags.

![[Pasted image 20250522143432.png]]


### VEUs and General-Purpose CPUs
In modern CPUs, scalar and vector instructions are used in any combination and can be **intermixed in the same program**. Vector instructions are often automatically used by a **vectorizing compiler**. 

Some examples of vectorized instruction sets:
- **Streaming SIMD Extension (SSE)** of x86 architectures. Vector registers of 128 bits. Signed and unsigned vector operations working on 64, 32, 16 or 8 bits operands. **SSE3** extension includes instructions able to work in parallel on the scalar values within one input vector register only (e.g., reduce).
- **Advanced Vector Extension (AVX)** of Intel. Introduction of wider vector registers (256 bits). Other variants adopt even wider registers (512 bits or 1024 bits in some prototypes)
- **ARM Neon** is an extension of the **ARMv7** and **ARMv8** RISC instruction sets, which includes vector instructions using vector registers of 64 and 128 bits and more.
- **PowerPC** architectures provide the Vector Scalar Extension (**VSE**) instruction set to support vector operations efficiently.


# References