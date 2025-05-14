**Data time:** 14:42 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Map Parallelization

With a generic **linear mapping strategy** each worker operates on g=L/n contiguous elements of A and g=L/n continues elements of B. So each received input array is **[[Scatter|scattered]]** and each output array is **[[Gather|gathered]]** before transmitting it outside.

![[Pasted image 20250514145955.png]]

Map lowers the [[Ideal Service Time]] (increases **throughput**) and lower the **[[Communication Latency|latency]]** to compute a single input. 
### Comparison with Farm and Pipeline
The previous computations is **on stream** and it is **stateless**. [[Farm]] and [[Pipeline]] can be feasible solutions too.
- **Farm** (Let A(i)/B(i) be the i-th input/output array respectively)
![[Pasted image 20250514150922.png]]

- **Pipeline desing** using loop unfolding:
![[Pasted image 20250514151002.png]]

### Map + [[Multicast]]
A map parallelization is not always a scatting phase followed by parallel computation by workers and a gathering phase.

**Example**: Map data distribution might not be a scatter.
![[Pasted image 20250514153407.png | 600]]
# References