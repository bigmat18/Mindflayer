**Data time:** 19:04 - 11-05-2025

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Optimal Parallelism Degree

it is a very meaningful and general concept. For stream-based computations, it expresses the **minimum parallelism degree** that a module shall use to achieve the **highest performance** (remove the bottleneck). In other words, the goal is to match the arrival speed using fewer resources as possible. It can be also defined with the following formula:
$$N_{opt} = \lceil T_{id-Q}/T_{A-Q}\rceil$$
Strong relationship between [[Utilization Factor|utilization factor]] and optimal parallelism degree. For example if $\rho=\frac{20}{5}= 4 > 1$ to improve the performance we need to multiply to 4.

### [[Optimal parallelism degree with single inputs]]
# References