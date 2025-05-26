**Data time:** 16:13 - 24-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Parallel and distributed systems. Paradigms and models]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Speedup

**Speedup** of a parallel program using $p$ processors ($S(p)$) is defined as the ratio of the [[Completion Time]] of the sequential time program $T_{c-seq}$ to the completion time obtained running with $p$ processors $T_c(p)$
$$
S(p) = \frac{T_{c-seq}}{T_c(p)}
$$
It measures how much faster a profram runs with more processors. The best speedup one can expect, varying the number of processors, is **linear speedup**.
- If there is **no parallelization overhead**, the maximum speedup with $p$ processors is $S(p) = p$
- There are execeptions, which are referred to as **super-linear speedup**

In general, because of the overhead introduced by the parallelization, the speedup is sub-linear with number of processors used.

### Superlinear Speedup
Can speedup be greater than p when using p processors? Yes, it can happen, although it is rare to see it. Two possibile reasons:
1. Unfair comparison with a naïve serial algorithm. E.g., the sequential version might not be compiled with optimization flags enabled. Can happen in some recursive algorithms (e.g., branch&bound algorithms)
2. Cache/memory effects, more processors mean more memory and larger caches, leading to fewer cache misses and page swapping
![[Screenshot 2025-05-25 at 21.33.45.png | 250]]     ![[Screenshot 2025-05-25 at 21.34.04.png|230]]

Unfortunately, quite often the speedup is sub -linear due to the parallel run -time overhead.
# References