**Data time:** 16:40 - 26-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Models of Computation]]

**Area**: [[Master's degree]]
# Work-Span Model

The he Work-Span model (also called **Work-Depth**) model provides more strict bounds than those offered by the [[Amdahl's Law]] and [[Scalability|Gustafson’s law]]. The program’s tasks form a DAG ([[Acyclic Computation Graphs|Directed Acyclic Graph]]). A task is a unit of work, i.e., **arbitrary sequential code**, that can be executed in parallel (using threads or processes) with other program’s tasks.

A given graph’s task is ready to run iff all its predecessors in the graph have been completed. odel assumptions: 
- $p$ identical processors, each executing one ready task at a time
- the task scheduling is **greedy**, i.e., whenever there is a ready task (and an available  processor), the task is executed immediately

![[Pasted image 20250526165139.png]]

- $T_p$ is the time when executing with a greedy scheduler with $p$ processors
- $T_1$ is called  **work**; $T_{\infty}$ is called **span** (and also called **critical path**)
- In the example DAG we have $T_1 = 54$ and $T_{\infty} = 27$

![[Pasted image 20250526170235.png]]

Let's consider the **[[Speedup]]** to derive some interesting bounds:

![[Pasted image 20250526170211.png | 450]]

### Brent' Theorem 
Assume a parallel computer whose processors can perform a task in unit time with greedy scheduling of
tasks. Assume that the computer has enough processors to exploit the maximum concurrency in an algorithm containing $T_1$ tasks such that it completes in $T_{\infty}$ time steps:
- At each level $i$ of the DAG, there are $m_i$ tasks (ie, $\sum_{i=1}^n m_1 = T_1$)
- We may use $m_i$ processors to compute all results at leve $i$ in O(i)

**Brent’s theorem** states that a similar computer with fewer processors p can execute the algorithm with the following upper time limit:
$$
T_p \leq \frac{(T_1 - T_{\infty})}{p} + T_{\infty}
$$
Brent’s inequality establishes an upper bound on $T_p$

**Proof sketch**:
- **The DAG has n levels**, and on each level $i$ there are $m_i$ operations
- Consider that at each level there is a unit task to compute, ie $n = T_{\infty}$
- For each level $i$, the time taken by $p$ processor is:
$$
T^{i}_{p} = \bigg\lceil  \frac{m_i}{p} \bigg\rceil \leq \frac{m_i + p - 1}{p} \:\:\:\:\: \bigg\lceil \frac{x}{y} \bigg\rceil \leq \frac{x +y-1}{y}
$$
than considering all n levels
$$
T_p = \sum_{i=1}^n T_p^{i} \leq \sum_{i=1}^n \left( \frac{m_i + p - 1}{p} \right) = \frac{1}{p} T_1 + \frac{(p-1)}{p}T_{\infty} = \frac{(T_1 - T_{\infty})}{p} + T_{\infty}
$$
##### Implication of Brent's theorem
Considering the [[Speedup]], we have that $S(p) \leq \min\left( p, \frac{T_1}{T_{\infty}} \right)$. To get good speedup, $T_1$ must be significantly large than $T_{\infty}$. In this case $(T_1 - T_{\infty}) \approx T_1$ therefore considering Brent's inequality again we have
$$
T_p \approx \frac{T_1}{p} + T_{\infty} \:\:\:\: if \:\:T_{\infty} << T_1
$$
When designing a parallel algorithm, focus on reducing the **span**, because the span is the fundamental
asymptotic limit on scalability. Increase the work only if it enables a drastic decrease in span. Overall, we have:
$$
\frac{T_1}{p} \leq T_p \leq \frac{T_1}{p} + T_{\infty} \:\:\:\: S(p) \geq \frac{p}{1 + p \cdot \frac{T_{\infty}}{T_1}}
$$
Additionally, it can be proved that:
$$
S(p) = \frac{T_1}{T_p} \approx p \:\:\:\: \frac{T_1}{T_{\infty}} >> p
$$
It says that greedy scheduling achieves (almost) linear speedup if a problem is **overdecomposed** to create much more potential parallelism than the number of processors. 
- The excess parallelism is called the parallel slack, and is defined as $\frac{T_1}{T_{\infty}}$
- In practice, it has been observed that a parallel slack of at least 8-10 works well

However, remember that these formulas assume:
- **No parallel overhead**, i.e., the parallel code does $T_1$ total operations
- The memory bandwidth is not a limiting resource (i.e., we consider [[PRAM Model]]-like abstract machines
- The scheduler is greedy in scheduling tasks (i.e., no delays due to lock or synchronization in general)

# References