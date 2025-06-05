**Data time:** 17:10 - 03-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Parallel Programming with OpenMP]]

**Area**: [[Master's degree]]
# Introduction to OpenMP

**OpenMP** in an API for **platform-independent** shared-memory parallel programming in C, C++, and Fortran providing high-level parallel abstractions on top of low-level threading mechanisms. OpenMP extends C, C++ and Fortran programming languages with directives (i.e., `#pragma omp`  …),
a few library routines (e.g., `omp_get_thread_num()`), and environmental variables (e.g.,
`OMP_SCHEDULE, OMP_NUM_THREADS`).

It is **natively supported** by almost all compilers (GCC, Intel, Clang,…). It is used to exploit shared-memory parallelism on CMPs (Chip Multi-Processors). The compiler automatically generates the necessary multithreaded code and all needed synchronizations based on the directives specified in the source code.

```c++
#pragma omp parallel for
for(int i=0;i<N; ++i)
	foo(i);
```

OpenMP allows programmers to organize a program into serial and parallel regions. It provides synchronization constructs through a “lightweight” syntax (compiler directives). **OpenMP is NOT a parallelizing compiler** (i.e., it does not automatically parallelize sequential code!)
- Parallelism is explicit
- Avoiding data races and obtaining good speedup, is programmer’s responsibility

OpenMP does not require that single-threaded code be restructured for threading:
- **It preserves sequential equivalence**
- It enables incremental parallelization of sequential programs

OpenMP relies mainly on compiler directives
- If the compiler does not recognize a directive, it ignores it
- Parallelization possible using just a small number of directives (both for coarse-grain and fine-grain parallelism)

If the compiler is not instructed to process OpenMP directives (i.e., -fopenmp), the program will
execute sequentially. Runtime routines have default sequential implementations

### OpenMP Execution Model (Fork-Join)
Higher-level than C++/POSIX threads. Implicit mapping and load balancing of tasks. The execution starts with a single thread called **Master thread**. The Master thread creates a **team** (or pool) of Worker
threads to execute **parallel regions** where tasks are computed in parallel by Workers. 

At the end of a parallel region there is an implicit barrier synchronization; after the barrier only the Master thread continues the execution. The **thread pool is reused** for next parallel regions.

![[Pasted image 20250603171812.png | 350]] ![[Pasted image 20250603171829.png | 340]]

### OpenMP Memory Model 
OpenMP **provides a relaxed consistency model that is similar to Weak Ordering**. It allows the reordering of accesses within a thread to different variables. Each thread has its temporary view of the memory (induced by machine registers, cache, etc.). Additionally, each thread has its **threadprivate** memory that cannot be accessed by other threads

The **flush** directive is used to enforce consistency between a thread’s temporary view of memory and the primary memory (or between multiple threads’ views of memory)
```c++
#pragma omp flush [variable list]
```
Use explicit flushes with care since they can evict cached values. Subsequent accesses might reload data from memory. **Flush operations are implied by all synchronization operations in OpenMP**, e.g., barrier, at entering/exiting to/from a parallel region, at the exit from workshare regions, etc. …

# References