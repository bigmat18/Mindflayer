**Data time:** 00:37 - 13-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Unpack-Compute-Pack Pattern

This is a Pattern to applay stream parallelism in non-stream use cases. [[Pipeline]] and [[Farm]] can be applied only if a large sequence of inputs (data stream) is present. Sometimes a computation working on a **single** (large) input can be re-factored in order to have a stram-base version
###### Example
![[Pasted image 20250513004143.png]]

The internal **compute** stage can be further parallelized with a **[[Pipeline]]** (based on an internal decomposition of F) or using a **[[Farm]]** (if F is stateless). We must be careful with the ordering of results.

### Const Model
The stream-based version of the system can be studied as a three-staged pipeline.

![[Pasted image 20250513004405.png]]

- The **[[Inter Calculation Time|inter-departure time]]** of the system is $T_{\Sigma} = \max\{L_{com}, T_{id-com}\}$
- The **[[Relative Efficiency]]** is $\epsilon_{\Sigma} = T_{id-upk}/T_{\Sigma}$
- The **[[Optimal Parallelism Degree]]** of **compute** is $N_{compute}^{opt} = \lceil T_{id-comp}/L_{com}\rceil$

### Communication Grain
The can do an **observation**: merging more data elements in the same message ca be useful to amortize $T_{setup}$. The goal is **minimize the [[Completion Time]]** $T_{C-\Sigma} \sim m \cdot T_{\Sigma}$ where $L_{com} = T_{setup} + s \cdot T_{transm}$

Adding more elements $s > 0$ to each message increases the calculation time of the compute state $T_{id-comp} \approx s\cdot T_{\Sigma}$

![[Pasted image 20250513005156.png |  500]]

**Example**: $m = 10³, T_F = 10³ \tau, T_{setup} = 10³\tau, T_{transm} = 10²\tau$.
With $s=1m T_{C-\Sigma} \approx 10⁶\tau$. Assuming $a=10$ we achieve $L \approx 16, N_{opt} = 7$ and $T_{C-\Sigma} \approx 10^5\tau$
# References