**Data time:** 12:13 - 12-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Pipeline

To introduce the pipeline steam parallelism we use a running example. Pipeline parallelism requires to existence of a **stream of inputs** to compute. The original sequential program computes each input by executing a **sequential of functions** (stateless for the moment).

Functions are ordered by **read-after-write data dependencies**, that means the input of a function is the output computed by the previous one.

![[Pasted image 20250512121728.png]]
- $T_A$ is the [[Inter Calculation Time|inter-arrval time]] of inputs to P
- $T_{id-P} = \max\{T_{calc-P},L_{com}\}$ is the [[Ideal Service Time]] of P
- $T_{calc-P} \sim \sum_{i=1}^n T_{F_i}$ is the calculation time of P, approximately equal to the sum of the processing time of the $n>0$ functions in the code.
- $L_p = T_{calc-P} + L_{com}$ is the **[[Communication Latency]]** to process a generic input by P (from the instant when it arrives to when the corresponding output is produces onto the output stream)

###### Pipeline Definition
A **pipeline** is a computation graph of $n>0$ stages (each implemented by a sequential process). Stage i receives inputs from stage $i-1$ and produces outputs to stage $i+1$.

The number of stages is the **parallelism degree** of the pipeline. More inputs are simultaneously executed by the pipeline each transformed by a different stage (**temporal parallelism**).

![[Pasted image 20250512123821.png]]

**Bottleneck elimination conditions**:
1. Number of stages qual to the [[Optimal Parallelism Degree]]
2. Stages are **balanced** (same cost).
3. [[Communication Latency]] **overlapped** with the [[Inter Calculation Time|inter calculation]] in all stages
### Cost Model
Generic cost model for a pipeline of $n>0$ sequential stages. The fist stage receives inputs with [[Inter Calculation Time|inter-arrival time]] $T_A$, main results of the cost model for the different **performance metrics**

![[Pasted image 20250512124258.png|550]]

Variation of the cost model in the previous slide, where the first stage is also in charge of **generating the stream (internally)**. Not always the stream is natively present in a problem. Sometimes it can be generated:

![[Pasted image 20250512124635.png]]

- In this scenario the [[Ideal Service Time]] of the system is the inverse of the **generation [[Processing Bandwidth|bandwidth]]** of the source.
- The system archives the highest [[Relative Efficiency|efficiency]] if the **[[Processing Bandwidth|output bandwidth]]** matches the generation bandwidth, ie, the **first stage is a [[Bottleneck Parallelization|bottleneck]]**.
##### Completion Time
The **[[Completion Time]]** of the whole stream (composed by $m>0$ dinstinct inputs) by a pipeline with $n>0$ stages is the sum of two components:

- **Initial filling transient** phase proportional to the number of stages.
- **Steady-state phase** proportional to the stream length.
The temporal diagram of a pipeline with four stages is the follow:

![[Pasted image 20250512131842.png | 500]]

The first block is the initial filling transient $(n-1)\cdot T_{D-\Sigma}$, and the second is the steady-state phase $m \cdot T_{D-\Sigma}$. The sum of this two parts is the completion time.

### Load Imbalance
The [[Optimal Parallelism Degree]] $N_{opt}$ is the minimum parallelism to eliminate the bottleneck. To be exploited with a pipeline parallelization, we should be able to identify $N_{opt}$ different functions in the sequential code all having the same calculation time.

Example of **load imbalance**:

![[Pasted image 20250512131157.png | 550]]
We have the [[Utilization Factor]] greatest than 1, for this we have a bottleneck, we weasted resources because the load is not balanced.

### Loop Unfolding
Pipeline is a white-box approach, ie, the parallel program should know the exact semantics and structure
of the sequential code to identify functions. Not always functions are so clear in the code, but they should be extracted using specific techniques. One of them is **loop unfolding** like in the example below:

![[Pasted image 20250512132339.png]]

**N-loop unfolding** where $n$ should be equal to $N_{opt} = \lceil T_{id-P}/T_A\rceil$. So doing $T_{calc-S[*]} \sim L/N_{opt} \cdot T_F$

In sintesi in this approach we split an array in $n$ part with $n = N_{opt}$ and we compute each part in a partition we the same operations. This produce a balanced stages.

### Stateful Pipeline
Until now we have seen pipeline stateless, we can also have stateful pipeline, if the state is partitioned and each partition is used bu one stage only (e.g. loop unfolding).

We can also have **inter-stage dependencies**, if the tate is updated by stage $i$ and read by stage $j$ where we have  $j > i$. For example

![[Pasted image 20250512133312.png]]

### Pipeline with Hazards
More complex scenarios is when the state is read by stage $i$ and updated by stage $i+1$. 

**Example**: P is the probability that he computation on the next input on stage $i$ uses the state produced by a stage $j$ where $j > i$:

![[Pasted image 20250512134342.png | 550]]

- The G computation plus the round-trip [[Communication Latency]] $S[0]$ <-> $S[1]$ are considered in the ideal service time of S[0] with probability p.
- The pattern arises in several important use (ess **processor's micro-architecture**)
# References