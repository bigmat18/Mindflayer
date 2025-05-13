**Data time:** 00:56 - 13-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Dataflow

Dataflow is a low-level parallelism exploitation paradigm. **Dataflow analysis** is a procedure where different tasks is a sequential program are analyzed to discover whether they can be run in parallel or not. It is used by **[[Level-based view for parallel computing|compilers]]** to apply different kinds of optimizations.
###### Task
A piece of code like a **function**, a **statement**, or even at the smallest granularity a task might be an **instruction**.

![[Pasted image 20250513010246.png | 600]]

How we can check if $F_1;F_2$ is equivalent in terms of computed results to $F_1||F_2$? The sufficient conditions are the so-called **[[Bernstein Conditions]]**

We build a **partial-order graph** where verticies are tasks and arcs are dependencies (i.e., dataflow graph). This can be a **physical execution graph** with a vertices executed by processes and arcs implemented by [[Channels in Message Passing|channels]]. Alternatively it can be **interpreted** by a different stream-parallelism paradigm ([[Farm]] with feedbacks). For example:

![[Pasted image 20250513011606.png | 600]]
##### Dataflow Example
Consider an example of a sequential program described by the execution of five functions in program order. We assume that each function cost $T_F = t, T_A = t/2, L_{com} = t/10$

![[Pasted image 20250513012447.png | 550]]

- P is **[[Bottleneck Parallelization|bottleneck]]** and need to be paralellized
- P computations is **stateless** since the internal state variables s and z are used on **read-only mode**
- Possible parallelization paradigms: **[[Farm]]**, **[[Pipeline]]** and Dataflow.

Now analyses 3 possibile parallelization solutions:
- **Parallelization 1**: the computation has a read-only state so, state variables can be replicated and we can use **[[Farm]]**.

![[Pasted image 20250513012813.png| 500]]

- **Parallelization 2**: We can arrange a **[[Pipeline]]** of five stages and input/output are properly propagated.

![[Pasted image 20250513013359.png | 500]]

- **Parallelization 3**: We can discover data dependencies among the five functions and design a **dataflow graph** (physical only in this example).

![[Pasted image 20250513013525.png | 550]]

### Read-Write State
It is also possible that state variables are affected by **[[Bernstein Conditions|write-after-read]] dependencies**. This may generate interesting use cases when the dataflow paradigm is applied on streams.

**Example**: $T_A = 4t, T_{F1} = 4t T_{F2} = 8t$

![[Pasted image 20250513014057.png | 500]]

### Synchronous Dataflow
**Synchronous dataflow** is an extension of the dataflow paradigm where each task/vertex of the graph has constant **consumption** and **production rates**.
###### Example
![[Pasted image 20250513014532.png | 350]]

- A task **fires** if the right number of messages is present in each of its input streams.
- In the example, the **firing sequence** ABCC clears all the queues of the dataflow by coming back to the initial condition.

The **consistency problem** of synchronus dataflow graphs consists of determinig if, for the given graph, there exists a firing sequence (**periodic schedule**) of all vertices of the graph that returns to the initial condition. The graph is **consistent** if such a periodic schedule exists.

In the example above:
![[Pasted image 20250513015936.png]]

- A period schedule does not exists
- Messages accumulate indefinitely in the queue between A and C.
- No firing sequence clears all queues going back to the initial conditions.

##### Lee's Theorem
A connected sychronus dataflow graph with $N > 0$ vertices has a perioc schedule if and only if its **topology matrix** $\Gamma$ has rank $N-1$

A **topology matrix** $\Gamma \in \mathbb{Z}^{|E|\times|V|}$. The element (a,b) is equal to the number of messages placed/consumed in the buffer of edge $a$ after the execution of vertex $b$. If $a$ is an input edge for $b, (a,b)$ is negative.

![[Pasted image 20250513020456.png | 600]]
# References