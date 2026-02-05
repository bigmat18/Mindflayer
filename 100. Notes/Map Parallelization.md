**Data time:** 14:42 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# Map Parallelization

A **map** is a computing pattern in which a single function F can be applied independently to each 
element of an input **collection**, producing an output collection **of the same cardinality**. The parallelism is exploited by using a pool of Workers, each working independently on a partition of the input collection and producing a partition of the output collection
- The map paradigm **decreases the [[Communication Latency]]** to compute one input collection
- If there is a stream of collections, it **lowers the service time** and increases **throughput**

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

### Finding the number of Workers
$T_c^{map}(n,k)$ depends on $k$ so what is the optimal value of $k$ that minimizes the completetion time?
Considering the linear communication cost model, i.e., $T_{comm}(n) = t_0 + n \times s$ and assuming that $T_{split(n,k)} \approx 0$ and $T_{gather(\frac{n}{k})} \approx 0$ we have:
$$
T_c^{map}(n,k) = k \cdot T_{comm}(n/k) + \frac{n}{k}\cdot T_F + T_{comm}(n/k) =
$$
$$
(k+1) \cdot T_{comm}(n/k) + n/k \cdot T_F = (k+1) \cdot (t_0 + \frac{n}{k}\cdot s) + \frac{n}{k}\cdot T_F
$$
$$
\frac{d}{dk}T_c^{map}(n,k) = o \text{ for } k_{opt} = \bigg \lceil \sqrt{\frac{n}{t_0} \cdot (T_F + s)} \bigg \rceil
$$
$k_{opt}$ is usually a very large number because $n$ is large and $t_0$ is very small

### Map cost model leveraging a Parallel Filesystem
Let’s consider again the sequential program
![[Pasted image 20260205022514.png]]

Since all tasks are available altogether and they are independent, we can consider them a collection, and therefore we can apply the Map pattern. The input file can be split into **k** sub-collections, and each Worker can independently read and work on one sub-collection.

Sub-collections must be merged into a single collection
- Often, when the total size of the output collection is known, a **sparse file** is used as the output collection so that each Worker can write independent regions without conflict

![[Pasted image 20260205022615.png | 400]]

In this case, we use only k resources (not k+2 as in the farm)
$$
T_c^{map}(n,k) = T_{split} + \frac{n}{k} \cdot T_{F+G} + T_{merge}
$$

Potential advantage of a **parallel filesystem (parallel I/O)**
- If the data is already distributed, the splitting overhead is minimal
- Splitting and merging are logical operations that do not actually move data
- In this scenario, $T_c^{map}(n,k) \approx \frac{n}{k} \cdot T_{F+G}$ which can be significantly lower than that obtained from centralized I/O
# References