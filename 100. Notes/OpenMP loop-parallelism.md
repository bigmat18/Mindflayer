,**Data time:** 23:41 - 03-06-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Parallel Programming with OpenMP]]

**Area**: [[Master's degree]]
# OpenMP loop-parallelism

When a thread reaches a **parallel** directive, it creates a team of threads and becomes the Master of the team (the master has ID 0). Each thread of the pool computes the code of the parallel region (i.e., the structured block following). At the end of the parallel region there is an implicit [[Barriers]] (Additional [[Synchronization Basics|synchronization constructs]] exist and can override or refine the implicit barrier)
```c++
#pragma omp parallel [clause list]
```
Some commonly used clauses:
- `if (scalar expression)`: determines whether the directive creates threads
- `num_threads (integer expression)`: number of threads to create
- `private (variable list)`: specifies variable local to each thread
- `firstprivate (variable list)`: similar to private, the variables are initialized to variable value before the parallel directive.
- `shared (variable list)`: specifies that variables are shared among all threads in the region. This is the default visibility scope of variables!
- `default (data scoping specifier)`: default data scoping specifier for the given region. Common specifiers include `shared` and `none`
- `reduction (operator : variable list)`: specifies how to combine local copies of a variable in different threads into a single copy at the master at region exit

### OpenMP worksharing directives
Within the scope of a **parallel** directive, a worksharing directive specifies that the created threads should execute statements, blocks, loop iterations, and tasks cooperatively (i.e., in parallel). There are four **worksharing directives**:

- **for**: threads cooperatively execute loop iterations. It partitions loop iterations across threads Possible clauses in `[clause list]` :
	- private, firstprivate, lastprivate
	- reduction
	- schedule, nowait, and ordered
```
#pragma omp for [clause list]
```
At the end of the for, there is an implicit barrier. It requires the so-called Canonical Loop Form [doc](https://www.openmp.org/spec-html/5.0/openmpsu40.html)

- **sections**: threads cooperatively execute code blocks
- **single**: one thread executes the block, other threads wait (Implicit barrier as it ends, which can be suppressed with nowait)
- **workshare**: divide the execution of enclosed structured block  into separate units of work (Fortran only)

###### Example using `parallel` and `for`
i![[Pasted image 20250603234852.png | 600]]

### Variable sharing and privatization
Following the list od variables scope possible with openmp:
- **`shared(x`)**: all threads access the memory location where x is stored (this is the default)
- **`private(x)`**: each thread has its private copy of x. 
	- All local copies are not initialized. 
	- Local updates to x are lost when exiting the parallel region. At the end of the region, x retains its original value.
- **`firstprivate(x)`**: each thread has its private copy of x
	- All local copies of x are initialized with the value that x has before starting the parallel region
	- Local updates to x are lost when exiting the parallel region. At the end of the region, x retains its original value.
- **`lastprivate(x)`**: primarily used in for and sections. It has the same semantics as private(x), but at the end of the region, x retains the thread local copy value of the thread that executes the last loop iteration or the last section.
- `default(shared)` or `default(none)`: affects all variables not specified in other clauses. default(none) ensures that you must specify the scope of each variable in the parallel block,
###### Example 1
```c++
#include <cstdio>
#include <omp.h>

int main() {

    int a=0;
    int b=1;
    int c=2;
    int d=3;

#pragma omp parallel num_threads(8) private(a) shared(b) firstprivate(c)
    {
        a++;  // not initialized! Warning from compiler (-Wall)
        b++;  // this is not atomic
        c++;  // local copies initialized 
        d++;  // this is shared by default, non atomic increment
        std::printf("Hi %d a=%d, b=%d, c=%d, d=%d\n",
                    omp_get_thread_num(), a, b, c, d);
    }
    std::printf("Final values: a=%d b=%d c=%d d=%d\n", a, b, c, d);

}
```

![[Pasted image 20250603235806.png | 300]]

the behavior of `b` and `d` are not regular because the increment is not atomic and there isn't any syncrhonization mechanism to hanlde them.
###### Example 2
```c
int main() {

    int i;

#pragma omp parallel for lastprivate(i) num_threads(8)
    for(int j=0; j<16; ++j) {
        i = j;
        std::printf("Hi %d i=%d\n", omp_get_thread_num(), i);
    }
    std::printf("Final value: i=%d\n", i);

}
```

![[Pasted image 20250604000047.png | 250]]

it this case, we have 8 thread for 16 iteration (2 iteration per thread) every time we print the i value that the thread that compute the last iteration has.
### Nested Parallelism
Nested parallelism enabled by using the `OMP_NESTED` environment variable or `omp_set_nested(1)`. In general **nested parallelism is rarely used**.

- Nested parallelism is not enabled by default
- E.g., OMP_NESTED=true ./myprog
- `omp_get_num_threads()` returns the size of the innermost pool to which the calling thread is part of (NOTE: if no thread pool is active, it returns 1)
- `omp_get_max_threads()` returns the maximum number of threads that can be created

![[Pasted image 20250604000637.png]]
###### Example
```c
// execute with OMP_NESTED=true ./omp_nested

int main() {

    //omp_set_nested(1);

#pragma omp parallel num_threads(3)
    {
        std::printf("Level 0 - (). Hi from thread %d of %d\n",
                    omp_get_thread_num(),
                    omp_get_num_threads());       
        int parent = omp_get_thread_num();
#pragma omp parallel num_threads(2)
        {
            std::printf("Level 1 - (%d). Hi from thread %d of %d\n",
                        parent,
                        omp_get_thread_num(), omp_get_num_threads());       
            int parent = omp_get_thread_num();
#pragma omp parallel num_threads(1)
            {
                std::printf("Level 2 - (%d). Hi from thread %d of %d\n",
                            parent,
                            omp_get_thread_num(), omp_get_num_threads());
            }
        }
    }
    return 0;
}
```


### `reduction` clause for `parallel` directive
It specifies the **binary associative operator** to use to combine local copies of a variable in different threads into a single copy at the master when threads exit. Reduction operators: `+, *, |, ^, &&, ||, min, max` (**Identity values**: 0, 1, 0, 0, 1, 0, INT_MAX, INT_MIN (or ±FLT_MIN))

```c
#pragma omp parallel reduction (op : variable list)
```

One private copy of each variable in the variable list is created for each thread Each copy is initialized with the neutral element of the reduction operator (e.g., 0 for +), then each thread executes the parallel region.

At the end of the region execution, the reduction operator is applied to the last value of each local reduction variable, and the initial value of the reduction variable it had before entering the parallel region.

```c++
int x=5;
int y=1;
#pragma omp parallel reduction(* : x) \
				     reduction(+ : y) num_threads(3)
{ // <- spawning 3 threads, x set to 1, y set to 0
	x += 3;
	y += 3;
} // <- joining threads

// the value of x here is 320 ( (1 + 3)3 × 5 )
// the value of y here is 10 (3 × 3 + 1 )
```

###### Example: estimate of $\pi$
Mathematically, we know that:
$$
\int^{1}_0 \frac{4.0}{(1 + x²)}dx = \pi
$$
t is possible to approximate the integral as a sum of the **N** rectacgles in $[0,1]$ where each rectangle has width $\Delta x$ and height $F(x_i)$ computed at the middle of interval $i$:
$$
\pi \approx \sum_{i=0}^{N} F(x_i) \times \Delta x
$$
Another option is to compute N steps of the following sum:
$$
\pi = 4\sum^{+\infty}_{i=0} \frac{(-1)^k}{(2k+1)}
$$
![[Pasted image 20250604002003.png | 180]]

```c++
int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage is: " << argv[0] << " num_steps\n";
        return -1;
    }

    uint64_t num_steps = std::stol(argv[1]);
    long double x = 0.0;
    long double pi = 0.0;
    long double sum = 0.0;
    long double step = 1.0 / num_steps;

    double start = omp_get_wtime();

#pragma omp parallel for private(x) reduction(+:sum)
    for (uint64_t k = 0; k < num_steps; ++k) {
        x = (k + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;

    double elapsed = omp_get_wtime() - start;

    std::cout << "Pi = " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << pi << "\n";
    std::cout << "Pi = 3.141592653589793238 (first 18 decimal digits)\n";
    std::printf("Time %f (ms)\n", elapsed * 1000.0);
    return 0;
}
```

- `x` must be private
- `sum` (i.e., the reduction variable) is automatically privatized

![[Pasted image 20250604002330.png | 350]]
### Mapping iterations to threads
The **`schedule`** clause of the **`for`** directive enables different loop iterations distribution policies (iterations
scheduling). The default schedule is implementation dependent. In GCC it is static. Four classes:
###### `static`
work assigned to threads statically at compile time in a cyclic fashion. If chunksize is not set, then $\big\lceil \frac{\#iterations}{nthreads}\big\rceil$. The iteration space is divided into pieces of size chunk

```c
#pragma omp parallel for [schedule policy] num_threads(5)
for (uint64_t i = 0; i < 16; i++) {
	...
}
```

![[Pasted image 20250605112240.png | 400]]

- **`schedule(static)`**
![[Pasted image 20250605112333.png | 400]]

- **`schedule(static,1)`** cyclic distribution
![[Pasted image 20250605112351.png | 400]]

- **`schedule(static, 2)`** block-cyclic distribution with c=2
![[Pasted image 20250605112410.png | 400]]

###### `dynamic`
Iterations assigned dynamically at run time at a granularity of `chunksize` (default `chunksize` is 1). When a thread completes one chunk another one (if available) is assigned to it.
- When a thread finishes executing a chunk, a new chunk is assigned by the runtime system
![[Pasted image 20250605112614.png | 600]]
- Logically, it is a master-worker computation paradigm where the master dispatches tasks to ready Workers
- The dynamic schedule requires more work at runtime (thus more overhead than the static schedule), but it **guarantees better workload balancing among Workers** if there are unpredictable or highly variable work per iteration.
- The `chunksize` is another **critical factor**, the optimal value is system- and application-dependent

###### `guided`
iterations assigned dynamically at run time at a granularity that is exponentially reduced with each dispatched piece of work up to `chunksize` (the default minimum `chunksize` is 1). The actual implementation depends on the compiler.
- In the **guided schedule**, the `chunksize` is a lower bound. Bigger chunks are assigned at the beginning, and as chunks are completed, the size of the new chunks decreases up to `chunksize`

###### `runtime`
The scheduling policy depends on the `OMP_SCHEDULE` environment variable
###### `auto`
delegates the decision of the scheduling to the compiler and the runtime system


###### Example: All-pairs distance Matrix
Results obtained running the all_pair.cpp implementation for the MNIST dataset on the cluster front-end node. Sequential time: 704s. `OMP_NUM_THREADS`=40

![[Pasted image 20250605112902.png | 300]]

###### Example: Mandelbrot Set
Each point of the Mandelbrot Set can be computed independently. To increase the computation granularity, it is possible to parallelize the outer loop (computing all pixels in a row). By using the clause **`schedule(runtime)`** we can play with different iteration scheduling policies by setting `OMP_SCHEDULE` variable.

```c++
int main(int argc, char *argv[]) {
  DISPLAY(gfx_open(XSIZE, YSIZE, "Mandelbrot Set"));
#if defined(NO_DISPLAY)
  volatile int fake = 0;
#endif
  TIMERSTART(mandel_seq);
#pragma omp parallel for schedule(runtime)
  for (int y = 0; y < YSIZE; ++y) {
    const double im = YMAX - (YMAX - YMIN) * y / (YSIZE - 1);
    // temporary buffer to store the pixel values for the current line
    std::vector<int> results(XSIZE);
    for (int x = 0; x < XSIZE; ++x) {
      const double re = XMIN + (XMAX - XMIN) * x / (XSIZE - 1);
      results[x] = iterate(std::complex<double>(re, im));
    }
#if defined(NO_DISPLAY)
    fake += results[0];
#else
#pragma omp critical
    drawrowpixels(y, results);
#endif
  }
  TIMERSTOP(mandel_seq);
  DISPLAY(std::cout << "Click to finish\n"; gfx_wait());
}
```

`#pragma omp critical` marks a region to be a **critical section**. Threads in the team access the region in mutual exclusion.

### The `collapse` clause
It specifies how many loops in a loop nest should be collapsed into one large iteration space and then divided according to the **schedule** clause. The main objective is to increase the iteration count.
```c
#pragma omp parallel for schedule(static) \
num_threads(5) collapse(2)
for (int i = 0; i < 4; ++i)
	for(int j=0; j < 5; ++j)
		do_something(i,j);
```

![[Pasted image 20250605114035.png | 400]]

![[Pasted image 20250605114047.png | 500]]


# References