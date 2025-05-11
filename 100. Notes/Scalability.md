**Data time:** 18:40 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Scalability
### Strong Scalability
Scalability can also call **strong scalability**. It is a metric representing how much the parallel version is faster than sequential one. It is computed is the following way:
$$S(n) = T_{C-\Sigma}(1)/T_{C-\Sigma}(n)$$
It can also computed using the [[Inter Time|inter-departure time]] or [[Processing Bandwidth|bandwidth]]. We compute the execution time with parallelism 1 against the execution time with parallelism n by keeping the problem size fixed.
##### Amdahl's Law
The Amdahl's law is one of the reason why scalability might not be ideal. T is the execution time, 1-P is the serial fraction.

![[Pasted image 20250511192219.png | 550]]

A const part (dark blue) can not be parallelized for this with a n=2 we have not exaclty 2 times faster situations. 
### Weak Scalability
##### Gustafson's Law
It assumes the parallel part scales linearly with the amount of resources, while the serial part does not increase with the problem size.

**Weak scalability** is:
$$S'(n) = T_{C-\Sigma}(1, n\cdot w)/T_{C-\Sigma}(n,n\cdot w)$$
where $T_{C-\Sigma}(1, n\cdot w)$ is the execution time with parallelism 1 and problem size n-times greater than the reference one w.

![[Pasted image 20250511193325.png | 550]]

If you increase the problem size, the serial part remain the same (dark blue) and the parallel part grows. The time is same time but with growing data.
### Scalability vs Speedup
Sometimes there is confusion in the literature about the use of two similar but different concepts: **scalability** and **speedup**. The difference is in the choice of the sequential implementation to be used as the baseline for comparison:

- **(Strong/Week) Scalability**: Requires that the sequential implementation is equivalent to the parallel implementation with n=1
- **Speedup**: Requires that the sequential implementation is the best one available (maybe also using a different sequential algorithms, not necessarily the best one suitable to be paralyzed)

![[Pasted image 20250511194129.png| 300]]
# References