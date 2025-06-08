**Data time:** 23:35 - 03-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Parallel Programming with OpenMP]]

**Area**: [[Master's degree]]
# OpenMP primitives and enviroment variables

C and C++ compilers use **`#pragma omp`**  prefix:
- Pragmas are preprocessor directives enabling behaviors that are not part of the language
- Compilers that do not support **`#pragma omp`** ignore it
- Fortran uses significant comments **`!$omp, c$omp, *$omp`** prefixes
```c++
prefix directive [clause list]
```

**C++ Example**: `#pragma omp parallel num_threads(8)`
**Fortran**: `!$omp parallel num_threads(8)`

Most OpenMP constructs apply to the **structured bloc**k following the directive:
- A structured block is a block of statements with **one entry point and one exit point** (returning or jumping inside a parallel block is not allowed)
- If the structured block i**s not explicitly marked** (using braces {}), **is taken the next single statement**

###### Example "Hello World"
![[Pasted image 20250603173347.png]]
- The `-fopenmp` compiler flag enables OpenMP support.
- The environment variable `OMP_NUM_THREADS` can be used to set the default number of threads in the teams that will be created for the parallel regions. In this example if it is not set, the number of threads in the team is equal to the number of available cores

A structured block can be a single function call (in this example a C++ lambda). The `-fopenmp` compiler flag defines the preprocessor variable `_OPENMP`, can be useful to include/exclude code at compile time. The clause **`num_threads`** is used to set the number of threads to create for a given region

![[Pasted image 20250603173601.png | 450]]

The if clause is evaluated and if its expression evaluates to true, the parallel construct is enabled with num_threads thread, otherwise is ignored. To measure the time, OpenMP provides a library routine **`omp_get_wtime()`**, which returns wall clock time in seconds.

![[Pasted image 20250603174931.png | 450]]

### Worksharing `sections` directive
The sections directive enables specification of [[OpenMP task-parallelism|task parallelism]] 

![[Pasted image 20250605115423.png | 350]]

The sections directive enables specification of task parallelism:
- Each **section** is executed once by one thread
- Different **section** may be executed by different threads
- There is an **implicit barrier** at the of of a **sections** directive, unless the **nowait** clause is used


```c++
#pragma omp parallel num_threads(2)
{
	#pragma omp sections
	{
		#pragma omp section
		{
		task1();
		}
		#pragma omp section
		{
		task2();
		}
		#pragma omp section
		{
		task3();
		}
	} // <- implicit barrier
} // <- implicit barrier
```

The execution order of tasks is not known. The assignment of tasks to thread is non-deterministic.

![[Pasted image 20250605115717.png | 150]]

```c++
int main(void) {
    const int N=128;
    float a[N], b[N], c[N], d[N];

#pragma omp parallel for num_threads(4)
    for (int i=0; i < N; i++) {
        a[i] = i * 1.0;
        b[i] = i * 3.14; 
    }

#pragma omp parallel num_threads(4)
    {
#pragma omp sections nowait
        {
#pragma omp section
            {
                std::printf("Thread %d executing sec1\n", omp_get_thread_num());
                for (int i=0; i < N; ++i)
                    c[i] = a[i] + b[i];
            }
#pragma omp section
            {
                std::printf("Thread %d executing sec2\n", omp_get_thread_num());
                for (int i=0; i < N; ++i)
                    d[i] = a[i] * b[i];
            }
        }  // <- no barrier here
    }  // <- implicit barrier here

    float r=0.0;
#pragma omp parallel for reduction(+ : r)
    for (int i=0; i < N; ++i)
        r += d[i] - c[i];

    std::printf("%f\n", r);
}
```

Note the **`nowait`** clause for the **sections** this. It means that at the end of the structured block, there is no implicit barrier.

### OpenMP synchronization constructs

- **`#pragma omp barrier`**: All threads in the active team must reach this point before they can proceed

- **`#pragma omp single [clause list]`** structured block: Mark a parallel region to be executed **only by one thread** (the first one reaching it, others skip the region). **NOTE**: like the other worksharing directives, there is an **implicit barrier at the end** of the block.

- **`#pragma omp master`** structured block: Mark a parallel region to be executed **only by the Master thread** (the one with ID 0). There is **no implicit barrier** at the end of the block.

- **`#pragma omp critical`** structured block: Mark the block to be a critical section. All threads will execute the critical section one at a time.

- **`#pragma omp ordered`** structured block: In loops with dependencies, ensures that carried dependencies do not cause data race It states that the structured block (within a loop) must be executed in sequential order.

- **`#pragma omp atomic`**: Only one thread at a time updates a shared variable

```c
#pragma omp parallel
{
	work_par();
	#pragma omp master
	work_seq();
	#pragma omp critical
	work_critical();
	#pragma omp barrier
	work_par();
} // <- implicit barrier
```

![[Pasted image 20250605114622.png | 350]]

### Thread Affinity in OpenMP
**Thread affinity** (or pinning or binding) controls how threads are mapped to HW resources (CPU cores and sockets). Keeping a thread on a specific core helps **maintain cache locality and reduce memory latency**, particularly on NUMA architectures, and **minimizing context switches overhead**

OpenMP Affinity management:
- **`OMP_PROC_BIND`** controls whether threads are allowed to migrate between cores. Values can be true, false, close, spread, and master
```
OMP_PROC_BIND=spread ./omp_affinity
```
- **`OMP_PLACES`** defines a list of «places» (e.g., cores, sockets) where threads can execute
```
OMP_PLACES=“cores” ./omp_affinity
OMP_PLACES=“{0:4, 8:4:2}” ./omp_affinity
```

Affinity is particularly beneficial for memory-bound and irregular workloads. Pay attention to the **first touch** allocation policy. **First touch with threading**: array elements are allocated in the memory of the NUMA node that contains the core that executes the thread that initializes the partition.

### SIMD parallelism in OpenMP
It directs the compiler to vectorize for loops. It is a hint to the compiler that loop iterations are independent. Clauses such **`simdlen`**, **`linear`** and **`reduction`** help control and fine-tune the vectorization
```
#pragma omp simd [clause list]
```
Can be combined with [[OpenMP loop-parallelism|parallel for]] and [[OpenMP task-parallelism|taskloop]]:
![[Pasted image 20250605131401.png]]

### Accelerators support (GEMM example)
OpenMP uses the **`target`** construct to offload execution to a target device Support for accelerators (e.g., [[Graphical Processing Units (GPU)|GPUs]]). For example
```
OMP_TARGET_OFFLOAD=MANDATORY/DISABLED ./myprog
```
forces OpenMP to run/not run the code on the device

Data moved implicitly between the host and the target device when entering and exiting the structured block. 
- Only for scalar, arrays, and structs with complete types resident in the stack of the task that encounters the target construct
- Data allocated on the heap must be explicitly copied

The clause **`data`** manage explicit data transfer, for example:
```
#pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
```

```c
int main() {
    // assume square matrices for simplicity
    constexpr int N = 4096;
    static float A[N * N];
    static float B[N * N];
    static float C[N * N];

    init(A, N * N);
    init(B, N * N);

    TIMERSTART(mm_naive1_openmp);
    // Offload the matrix multiplication to the GPU
#pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    TIMERSTOP(mm_naive1_openmp);

#if 0
    float *check = new float[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            check[i * N + j] = 0;
            for (int k = 0; k < N; k++)
                check[i * N + j] += A[i * N + k] * B[k * N + j];
            if (std::abs(C[i * N + j] - check[i * N + j]) > 1e-3f) {
                std::cout << "Result error: " << C[i * N + j] << " expected " << check[i * N + j] << std::endl;
                abort();
            }
        }
    }
    std::cout << "Result is ok!" << std::endl;
    delete[] check;
#else
    (void)C;
#endif
}
```
# References