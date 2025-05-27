**Data time:** 16:44 - 26-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Parallel and distributed systems. Paradigms and models]] [[Parallelization methodology and metrics]]

**Area**: [[Master's degree]]
# Amdahl's Law

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
# References