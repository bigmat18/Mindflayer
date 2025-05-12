**Data time:** 12:57 - 12-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]]

**Area**: [[Master's degree]]
# Bottleneck Parallelization

For first we have identified the **bottleneck condition** ($\rho > 1$) and next design a proper parallelization. The **goal** in a **stream-based scenarios** the parallelization goal is to reduce the ideal-service time of the parallelized module s.t. it matches the inter-arrival time, i.e.
$$T_{id-Q}(n) = T_{A-Q}$$
$T_{id-Q}(n)$ is the [[Ideal Service Time|ideal service time]] with a parallelism of n. The ideal case (without performance degradentions) the [[Optimal Parallelism Degree|right parallelism degree]] use the optimal one:
$$T_{id-Q}(n) = \frac{T_{id-Q}(1)}{n} \:\:\:\:\:N_{opt} = \bigg\lceil\frac{T_{id-Q}(1)}{T_A}\bigg\rceil$$
In case performance degradation exist (e.g., load imbalance), the optimal parallelism degree might be insufficient, or no parallelism degree can remove the bottleneck with a given parallelization. In all cases, the chosen parallelization strategy shall preserve the **computation semantics**.

In the example above the computations is **[[HTTP|stateless]]** there is side effects in the function. Therefore, functional replication is a feasible parallelization strategy for the problem.

![[Pasted image 20250511161859.png]]

We have 4 replica of the same function Q. When an input arrive it goes to one of free replica. If all replica are busy the input wait to be processes. In this cases **we don't change [[Networks Metrics|latency]]**
# References