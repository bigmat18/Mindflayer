**Data time:** 16:46 - 24-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[SIMD on CPU]]

**Area**: [[Master's degree]]
# Programming with AVX2

Special compiler-supported functions that enable direct use of CPU-specific instructions (e.g., SIMD operations) in C/C++ code **instead of writing raw assembly**. They enable access to low-level HW instructions while maintaining portability and ease of use compared to pure assembly. Not all intrinsic functions map one-to-one to a single assembly instruction—some may be implemented using multiple instructions.

Why using **Intrinsics SIMD**: 
- Intrinsics are expanded inline eliminating function call overhead
- Enable direct access to vectorized operations for higher performance
- More readable and maintainable than raw assembly while providing similar performance 
- Allow programmers to manually optimize code where compiler auto-vectorization might fail 

![[Pasted image 20250524165216.png]]

###### Example
![[Screenshot 2025-05-25 at 16.51.01.png | 550]]

### Aligned and Unaligned Operations
Aligned and unaligned operations refer to how data is stored in memory relative to certain byte boundaries. Data is aligned when its **memory address is a multiple of a specific boundary** (e.g., 32 bytes for AVX). Aligned memory accesses are generally faster. For unaligned accesses, the CPU may need to perform extra work to handle data crossing cache line boundaries

- **Why both are necessary?** Not all data structures are naturally aligned, e.g., those with complex layouts. When data can be allocated aligned, it is better to use aligned operations for best performance. Programmers need to evaluate the convenience/flexibility of unaligned operations vs. performance gains.

- **How to allocate aligned memory?**
```c++
// intrinsics 
float *a=(float*)_mm_alloc(size, 32); 
_mm_free(a);

// C++17 
float *a = new(std::align_val_t(32)) float[size]; 
delete [] a;

// C11 
float *a=(float*)aligned_alloc(32, size); 
free(a);
```

### Fused Multiply Add (FMA)
FMA is a single instruction that performs a multiplication and an addition.
$$
FMA(A, B, C) = (A \times B) + C
$$
FMA is used for:
- Fewer instructions executed leads to better performance in critical loops (e.g., matrix multiplications) because often FMA can be executed in a single cycle.
- Many modern CPUs (and [[Graphical Processing Units (GPU)]]) have dedicated FMA units within the [[Vectorization Instructions|Vector Units]]
- Unlike separate operations, FMA performs one rounding at the end, enhancing numerical accuracy. When multiplication and addition are done separately, rounding errors can accumulate

![[Screenshot 2025-05-25 at 17.01.13.png | 400]]

### GEMM with AVX2
###### Textbook GEMM
```c
// Textbook matrix multiply with AVX2 and FMA 
void mm_avx(float * A, float * B, float * C, uint64_t M, uint64_t L, uint64_t N) { 
	for (uint64_t i=0; i<M; i++) {
		for(uint64_t j=0; j<N; j+=8) {// N multiple of 8 
			// init temporary vector 
			__m256 X = _mm256_setzero_ps();
			for (uint64_t k=0; k<L; k++) {
				// replicate A[i][k] across all 8 lanes 
				__m256 Aik = _mm256_broadcast_ss(&A[i*L+k]); 
				// load 8 contiguous float from 
				B[k][j…j+7] __m256 BV = _mm256_load_ps(&B[k*N+j]); 
				X = _mm256_fmadd_ps(Aik, BV, X); // FMA
			}
			// store the computed 8 values into C[i][j…j+7] 
			_mm256_store_ps(&C[i*N+j], X);
		}
	}
}
```

###### Transposed GEMM
```c
//Transpose-and-Multiply with AVX2 and FMA 
void avx_tmm(float * A, float * B, float * C, uint64_t M, uint64_t L, uint64_t N) { 
	for (uint64_t i=0; i<M; i++) {
		for (uint64_t j=0; j<N; j++) {
			__m256 X = _mm256_setzero_ps();
			for (uint64_t k=0; k<L; k+=8) { // L multiple of 8
				const __m256 AV = _mm256_load_ps(A+i*L+k); 
				const __m256 BV = _mm256_load_ps(B+j*L+k); 
				X = _mm256_fmadd_ps(AV, BV, X); // FMA
			}
			// horizontal sum reduction; from 8 floats to a single one
			C[i*N+j] = hsum_avx(X); 
		}	
	}
}
```

###### `hsum_avx` implementation
![[Screenshot 2025-05-25 at 17.24.07.png]]

###### Performance
![[Screenshot 2025-05-25 at 17.24.49.png | 500]]

### Divergence
```c
//Mapping a conditional Statement onto SIMD 
for (i = 0; i 0)
	if (u[i] > 0)
		w[i] = u[i]–v[i]; 
	else 
		w[i] = u[i]+v[i];
```

**Divergent branching**: When lanes evaluate a condition individually and take different control paths due to branches.

![[Screenshot 2025-05-25 at 17.37.30.png| 500]]

- All ALUs required to execute the same instruction (synchronously) or idle
- Many [[SIMD (Single Instruction, Multiple Data)]] processors handle divergence by serializing the execution of different branches
- This problem can be addressed using **predication**

##### Predication
With predication we can handle divergence in the following way:
- **Compute both paths**: in the example below we must compute both the multiply by 2 (mul) and the division by 2 (div)
- **Create a mask of elements** with all bits set to 1 if the computed value is less than 0.0f, and 0 otherwise
- Use the mask to select the result (**blending**) from mul or from div
- This way, all lanes (elements) go through the same sequence of instructions (there is no if statement)

![[Screenshot 2025-05-25 at 17.43.36.png | 250]]

The code became the following:
```c
// AVX version using predication (branch-free) 
void transform_scalar(const float* in,const float* out, size_t n) { 
	__m256 zero = _mm256_set1_ps(0.0f); 
	__m256 two = _mm256_set1_ps(2.0f); 
	
	// suppose n % 8 = 0 for simplicity
	for (size_t i = 0; i < n; i += 8) {
		// unaligned load __m256 
		v = _mm256_loadu_ps(&in[i]); 
		__m256 mul = _mm256_mul_ps(v, two); 
		__m256 div = _mm256_div_ps(v, two); 
		
		// create the mask considering elements LT zero 
		__m256 mask = _mm256_cmp_ps(v, zero, _CMP_LT_OS); 
		// if mask=1 take from mul otherwise take from div 
		__m256 blend = _mm256_blendv_ps(div, mul, mask); 
		// store the results back to memory 
		_mm256_storeu_ps(&output[i], blended); 
	}
}
```

![[Screenshot 2025-05-25 at 17.50.49.png | 400]]

### AoS and SoA layouts
To exploit the power of **[[SIMD (Single Instruction, Multiple Data)]] parallelism**, it is often necessary to **modify the layout of the data structures used**.

![[Screenshot 2025-05-25 at 17.52.44.png | 500]]

The case study (in the pictures) is a collection of n values representing 3D coordinates. Two different layouts:
- **AoS (Array of Structures)**: stores records consecutively in a single array
- **SoA(Structure of Arrays)**: uses one array per dimension. Each array only stores the values of the associated element dimension

![[Screenshot 2025-05-25 at 17.55.21.png|400]]

##### Vector Normalization with AoS

![[Screenshot 2025-05-25 at 17.56.19.png | 550]]

Vectorization of 3D vector normalization based on the AoS format is **relatively inefficient**:
- Vector registers would not be fully occupied. E.g., for 128-bit registers we use three out of four vector lanes
- Summing up the squares requires operations between neighboring horizontal lanes, resulting in only a single value for the inverse square root calculation
- Scaling to longer vector registers becomes increasingly inefficient

![[Screenshot 2025-05-25 at 18.01.47.png | 350]]

##### Vector Normalization with SoA

![[Screenshot 2025-05-25 at 18.06.01.png |550]]

In this version, during each loop iteration, eight vectors are normalized simultaneously.

![[Screenshot 2025-05-25 at 18.07.00.png | 400]]

To converto from AoS to SoA we can use **Vectorized Shuffling**:

![[Screenshot 2025-05-25 at 18.08.08.png | 500]]

AoS on the fly transposition into SoA and inverse transposition of results:

![[Screenshot 2025-05-25 at 18.09.07.png|500]]

In general, for a code style, Prefer SoA to AoS:
- SoA makes it easier for compilers to perform vectorized loads/stores
- Converting AoS to SoA might require restructuring the code and data, which can be non-trivial

### Loop iteration dependencies
Not all loops will be vectorized automatically, compilers can be conservative in many cases. For GCC, `-O3 -march=native -ffast-math` activate aggressive auto-vectorization on most loops. (Warning: `-ffast-math` can slightly change numerical results).

If auto-vectorization fails for critical loops, consider:
- Simplifying loop structure or remove (reduce) dependencies
- Using compiler hints/pragmas (e.g., `#pragma GCC ivdep` or `#pragma ivdep` for Intel’s compiler or `#pragma omp simd` if using OpenMP, `ivdep stads` for “ignore vector dependencies”)
- Employing intrinsics for manual vectorization if critical loops are not auto-vectorized. Auto-vectorization and Intrinsics code may coexist.

```c++
#pragma GCC ivdep 
for (auto i = 0; i < N; i++) { 
	// loop body that you claim has 
	// no dependencies 
}
```

With these dependencies, the compiler does not automatically vectorize the loop because it cannot safely reorder loop iterations:
- True dependencies **([[Bernstein Conditions|read-after-write]])**: An iteration uses data produced in a previous iteration
```c
for (int_i=1; i<N; ++i) {
	A[i]= A[i-1] + B[i];
}
```

- Anti dependencies (**[[Bernstein Conditions|write-after-read]]**): An iteration writes to a memory location read in a previous iteration
```c
for (int_i=1; i<N; ++i) {
	B[i] = A[i]; 
	A[i] = A[i+1];
}
```

- Output dependencies (**[[Bernstein Conditions|write-after-write]]**): Multiple iterations write to the same memory location
```c
for (int_i=1; i<N; ++i) {
	sum += A[i];
}
```

When dependencies may still allow vectorization: Reduction pattern, If the loop can be refactored to remove or isolate the dependencies

##### Aliasing
Auto-vectorization is often blocked by potential aliasing. Two (or more) pointers (or references) point to overlapping rengions of memory. **Runtime checks** are a strategy that compilers can use to enable vectorization.

Use compiler extension qualifier, e.g., `__restrict__` in C++ to tell the compiler that there is no aliasing.
```c++
// potential aliasing for a and b 
for (size_t i = 0; i < n; i++) 
	a[i] += b[i];

// runtime check to enable vectorization 
if ( a + n < b || b + n < a) { 
	// a and b do not overlap 
	// vectorized code 
} else { 
	for (size_t i = 0; i < n; i++) 
		a[i] += b[i]; 
}

// If you know there are no aliasing issues 
void add(float* __restrict__ a, const float* __restrict__ b, size_t n) { 
	for (size_t i = 0; i < n; ++i) 
		a[i] += b[i]; 
}
```
##### Loop unrolling
**Map reduction**: the `plain_main` function computes the maximum of a given array of floating-point numbers. This loop is not auto-vectorized by the compiler. We could use `#pragma omp reduction(max:max)` it requires `–fopenmp` compiler flag.

```c
float plain_max(float * data, uint64_t length) { 
	float max = -INFINITY; 
	for (uint64_t i = 0; i < length; i++) 
		max = std::max(max, data[i]); 
		return max; 
}
```

What do you expect if we unroll the loop by a factor of 2?
```c
float plain_max_unroll_2(float * data, uint64_t length) { 
	float max_0 = -INFINITY, max_1 = -INFINITY; 
	for (uint64_t i = 0; i < length; i+=2) { 
		max_0 = std::max(max, data[i+0]); 
		max_1 = std::max(max, data[i+1]); 
	} 
	return std::max(max_0, max_1); 
}
```


**General tips** for loop vectorization:
- In general, for code style, Prefer unit stride access in the innermost loop (e.g., i,k,j instead of i,j,k in GEMM): Efficient prefetching of data, loading of vector registers with fewer instructions, better cache locality
- The exit of the loop must not be data-dependent. Avoid conditional ‘break’.
- The loop count must be known at entry to the loop. The variable setting the number of iterations must remain constant for the duration of the loop
- Avoid `switch/return` statements inside the loop. The ‘if’ statement can be vectorized if it can be implemented as a masked assignment
- No function calls. If a function call can be inlined, this is okay. Intrinsics math functions (sin, sqrt, etc…) are allowed. Library functions inside the loop body may prevent vectorization
- Use aligned addresses to avoid compiler peels the accesses. If unaligned, the compiler can generate extra code (peeling) to handle misaligned data at the beginning/end of a loop
- Avoid dependencies between loop iterations, if possible
- Avoid pointer aliasing by using `__restrict__`


# References
[Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.htm)
[Intel® C++ Compiler Classic Developer Guide and Reference (Intrinsics section)](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/overview.html)