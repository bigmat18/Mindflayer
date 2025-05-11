**Data time:** 18:39 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Processing Bandwidth

#### Processing Bandwidth (Definition)
the inverse of the [[Parallelization methodology and metrics|ideal service time]] of process Q is its processing **[[Networks Metrics|bandwidth]]** (or **throughput**)
$$B_{id-Q} = 1 / T_{id-Q}$$


- **Ideal Processing Bandwidth** of Q
$$B_{id-Q}(n) = \bigg(\frac{T_{id-Q}(1)}{n}\bigg)^{-1}$$
- **Input Bandwidth** of Q
$$\lambda= (T_{A-Q})^{-1}$$
- **Output Bandwidth** of Q
$$B_{D-Q}(n) = \max\{B_{id-Q(n), \lambda}\}$$
Let assume that the ideal service time scales perfectly with the parallelism degree.

![[Pasted image 20250511163136.png | 500]]

If there is a bottleneck the ideal bandwidth is equal to output bandwidth. 
# References