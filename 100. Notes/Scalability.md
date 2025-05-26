**Data time:** 18:40 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Scalability
### Strong Scalability
Scalability can also call **strong scalability**. It is a metric representing how much the parallel version is faster than sequential one. It is computed is the following way:
$$S(n) = T_{C-\Sigma}(1)/T_{C-\Sigma}(n)$$
It can also computed using the [[Inter Calculation Time|inter-departure time]] or [[Processing Bandwidth|bandwidth]]. We compute the execution time with parallelism 1 against the execution time with parallelism n by keeping the problem size fixed.
##### Amdahl's Law
**Amdahl's law** describes the theoretical limit on the achievable scalability (and [[Speedup]]) when using multiple processors for a fixed problem size. The Amdahl's law is one of the reason why scalability might not be ideal. T is the execution time, 1-P is the serial fraction.

![[Pasted image 20250511192219.png | 550]]

A const part (dark blue) can not be parallelized for this with a n=2 we have not exaclty 2 times faster situations. It states that the scalability is fundamentally limited by **the fraction of sequential code** (the portion of the code that cannot be parallelized). 

![[Screenshot 2025-05-25 at 21.56.08.png | 350]]

Almost every program has a fraction of the code that cannot be parallelized. This fraction must be executed sequentially, even in a parallel implementation. Thus, Amdahl’s law establishes an upper bound on achievable speedup/scalability.

We now assume that the best possible speedup/scalability we can achieve is linear (no super-linear speedup). Then, we derive an upper bound for achievable speedup/scalability.

![[Screenshot 2025-05-25 at 21.58.24.png|350]]

Instead of using absolute execution times ($T_{ser}$ and $T_{par}$), we now use their relative fraction, f is the serial fraction whereas (1-f) is the parallelizable fraction. Substituting f in the previously derived upper bound, the result is Amdahl’s Law, the upper bound for the speedup only depends on f and p.

![[Screenshot 2025-05-25 at 22.00.57.png | 400]]

![[Screenshot 2025-05-25 at 22.01.27.png | 600]]

The sequential fraction of code, f, is a unit-less measure between 0 and 1. Amdahl’s Law can be used to predict the performance of parallel programs.

**Efficiency** under Amdahl's Law has the following formula and graph:

![[Screenshot 2025-05-25 at 22.13.38.png| 550]]

Why does Amdahl’s Law establish an unrealistic upper bound? Amdahl’s Law does not account for:
- **Communication cost**: even in the simplest parallel model, some communications or data movement are present
- **Overheads**: every parallel implementation introduces overheads due to synchronization and coordination
It is not true that we can arbitrarily increase p and always improve the speedup. A more plausible Amdahl’s Law would be something like:
$$
S(p) \leq \frac{1}{[f + O(p)] + \frac{(1-f)}{p}}
$$
Where O(p) is the additional serial fraction of time that includes any source of overhead, such as idle time, synchronization, and communication costs for running the application in parallel on p processors.

### Weak Scalability
Amdahl’s law only applies in situations where the problem size is fixed and the number of processors varies (strong scalability) • However, when using more PEs, we may also use larger problem sizes (weak scalability) • In this case, the time spent in the parallelizable part ($T_{par}$) may grow faster in comparison to $T_{set}$

**Scaled Speedup**, incorporates such scenarios when calculating the achievable speedup. We derive a more general law that enables us to study the scaling with respect to the problem’s complexity

**Derivation of Scaled Speedup Law**:
![[Screenshot 2025-05-25 at 22.23.18.png|500]]
##### Gustafson's Law
**Gustafson’s Law** is a particular case of the Scaled Speedup that can be used to predict the theoretically achievable speedup using multiple processors when the parallelizable part scales linearly with the problem size (Weak scaling), while the serial part remains constant.

It assumes the parallel part scales linearly with the amount of resources, while the serial part does not increase with the problem size.

![[Screenshot 2025-05-25 at 22.17.07.png|600]]

If we introduce a different form of Amdahl's Law we have the following formula:

![[Screenshot 2025-05-25 at 22.19.24.png | 450]]

Using different functions for $\gamma$ yields to the following two notable cases:
- $\gamma=1$ (ie $\gamma=\beta$) we have **Amdahl's Law**
- $\gamma=1$ (eg $\alpha=1; \beta=p$) i.e., the parallelizable part grows linear in p while the non-parallelizable part remains constant. We have **Gustafson’s law**:

![[Screenshot 2025-05-25 at 22.21.42.png| 350]]

**Weak scalability** is:
$$S'(n) = T_{C-\Sigma}(1, n\cdot w)/T_{C-\Sigma}(n,n\cdot w)$$
where $T_{C-\Sigma}(1, n\cdot w)$ is the execution time with parallelism 1 and problem size n-times greater than the reference one w.

![[Pasted image 20250511193325.png | 550]]

If you increase the problem size, the serial part remain the same (dark blue) and the parallel part grows. The time is same time but with growing data.

### Analysis case study
We have:
- **Input**: Array A of n numbers
- **Output**: $\sum^{n-1}_{i=0} A[i]$
- **Task**: Parallelize this problem by using $p$ **processing elements (PEs)**

Assumptions (Assumptions are unrealistic, they are simple enough to allow us to perform a basic analysis):
- **Computation**: Each PE can add two numbers stored in its local memory in 1 sec
- **Communication**: A PE can send data from its local memory to the local memory of any other PE in 3 sec (independently of the data size!)
- **Input and Output**: At the beginning of the program, the whole input array A is stored in PE #0. At the end, the result must be gathered in PE #0
- **Synchronization**: All PEs operate in a lockstep manner; i.e., they can either compute, communicate, or be idle (there is no computation-to-communication overlap)

Establish sequential runtime as a baseline (p = 1): For our example $T(1,n) = n-1 \:sec$. We are interested in:
- **Speedup**: How much faster can we get with p > 1 processors?
- **Efficiency**: Is our parallel program efficient?
- **Scalability**: How does our parallel program behave when the number of processors varies keeping the problem size fixed (strong scaling) and when the problem size changes (weak scaling)?

![[Screenshot 2025-05-25 at 21.40.40.png|200]]

Establish runtime for 2 PEs (p = 2) and 1024 numbers (n = 1024): T(2,1024) = 3 + 511 + 3 + 1 = 518 sec:
- **Speedup**: T(1,1024)/T(2,1024) = 1023/518 = 1.975
- **Efficiency**: 1.975/2 = 98.75%

![[Screenshot 2025-05-25 at 21.41.53.png|200]]

Instead with T(4,1024) = 3x2 + 255 + 3x2 + 2 = 269 seconds:
- **Speedup**: T(1,1024)/T(4,1024) = 1023/269 = 3.803
- **Efficiency**: 3.803/4 = 95.07%

![[Screenshot 2025-05-25 at 21.43.17.png| 200]]

Finally with T(8,1024) = 3x3 + 127 + 3x3 + 3 = 148 seconds
- **Speedup**: T(1,1024)/T(8,1024) = 1023/148 = 6.91
- **Efficiency**: 6.91/8 = 86%

![[Screenshot 2025-05-25 at 21.44.14.png|250]]

Timing analysis using $p = 2^q$ PEs and $n = 2^k$ input numbers:
- Data distribution: $3 \cdot q$
- Computing local sums: $n/p - 1 = 2^{k-q}-1$
- Collection partial results: $3 \cdot q$
- Adding partial results: $q$

![[Screenshot 2025-05-25 at 21.46.36.png | 350]]

**Strong scalability** analysis with n=1024:
![[Screenshot 2025-05-25 at 21.47.32.png|500]]

**Weak scalability** Analysis with n=1024xp:
![[Screenshot 2025-05-25 at 21.48.15.png|500]]

### Scalability vs Speedup
Sometimes there is confusion in the literature about the use of two similar but different concepts: **scalability** and **[[Speedup]]**. The difference is in the choice of the sequential implementation to be used as the baseline for comparison:

- **(Strong/Week) Scalability**: Requires that the sequential implementation is equivalent to the parallel implementation with n=1. We consider the time obtained executing the **parallel implementation of the algorithm on a single processor**.
- **Speedup**: Requires that the sequential implementation is the best one available (maybe also using a different sequential algorithms, not necessarily the best one suitable to be paralyzed)

![[Pasted image 20250511194129.png| 300]]
# References