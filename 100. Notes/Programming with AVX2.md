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


# References
[Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.htm)
[Intel® C++ Compiler Classic Developer Guide and Reference (Intrinsics section)](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/overview.html)