**Data time:** 18:38 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Relative Efficiency

The **Efficiency** of a parallel program parallelized using p processors $E(p)$ is defined as the ratio between the speedup $S(p)$ and the number of processors used $p$
$$
E(p) = \frac{S(p)}{p} = \frac{T_{t-seq}}{T_c(p) \times p}
$$
$T_c(p) \times p$ is also called **cost of parallization** (or simply **cost**). If $T_c(p) \times p > T_{c-seq}$ then the parallelization overhead (is, communication, synchronization, idleness) impairs parallel efficiency and performance.

The **relative efficiency** is a relative metric telling us how close/far is the effective performance from the ideal one. In the other words, it gives the general idea of the quality and effectiveness of the parallelization.

![[Pasted image 20250511183632.png | 550]]

In the figure above, we consider a **n-parallelization** of Q. The relative efficiency can be measured for a single module or w.r.t. whole system $\Sigma$
##### Efficiency and [[Processing Bandwidth|Bandwidth]]
Efficiency and bandwidth have similar shapes for the whole system $\Sigma$ while they are substantially different for a single module.

For any $n < N_{opt}$, Q is fully utilized and its bandwidth increases, while $n \geq N_{opt}$ the bandwidth is maximum and constant but the module'a efficiency decrease.

![[Pasted image 20250511185015.png | 550]]

When we arrive to the optimal Number of replica adding replica is not useful and the queue is less utilize.
# References