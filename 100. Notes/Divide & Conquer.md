---
Data: 2026-02-05T02:44:00
Tags:
  - note
  - youngling
Connection:
  - "[[Parallel and distributed systems. Paradigms and models]]"
  - "[[Structured Parallel Programming]]"
Area: "[[Master's degree]]"
---
# Divide & Conquer

**Divide & Conquer (DC)** is a data-parallel computing paradigm that breaks problems into smaller, 
independent subproblems, solves them concurrently, and combines the partial results. 
- **Divide**: partition the problems into smaller independent problems
- **Conquer**: compute each subproblem in parallel. The parallelism is exploited mainly (not necessarily only) in this phase
- **Combine**: Merge the partial solutions

![[Pasted image 20260205024527.png]]

Achieving efficient parallel DC requires addressing two challenges:
- **Scheduling the subproblems across processors to have good load balancing**. Dynamic scheduling strategies, such as dynamic work distribution, and work-stealing, are usually employed
- **Combining results without creating bottlenecks**

To avoid excessive overhead, a threshold value in the Divide phase is necessary
- A predefined problem size below which the divide phase stops
- It ensures that the overhead of recursive division and task distribution does not outweigh the benefits of parallel execution

Two primary options depending on if the workload is balanced or unbalanced among Workers
- **Balanced workload**. This means that the following assumptions hold (In this case, the DC implementation **is a [[Map Parallelization]] pattern** Divide+Scatter, then Conquer in parallel, then Gather + Collect):
	- The subproblems all have almost the same computational cost
	- It is possible to split the initial collection in at least k partitions

- **Unbalanced workload.**
	- Farm-based implementation with dynamic scheduling policy; the three steps are executed in pipeline.
	- Work-stealing-based implementation: coarse-grain Divide, then each process keeps dividing and storing in a local task queue. When the local task queue is empty, the Worker selects one of the processes at random and steals some work. More complex and costly termination condition

### Possible DC implementation
Let’s do the following assumptions:
- Input and output collections have the same size n
- The Divide phase produces a collection of subproblems {{S}} of cardinality d
- Workers implement a work-stealing algorithm during the Conquer phase (to balance the workload among Workers)
- Work-Stealing communications are overlapped with local computation (i.e., local conquer)
- The conquer phase for a Worker terminates after c failing attempts to steal from other Workers
	- The Worker moves to a global barrier waiting for all Workers terminate 
- The divide and Combine phase have a cost of $T_{divide}$ and $T_{merge}$ , respectively

![[Pasted image 20260205024945.png | 500]]

### Cost Model
![[Pasted image 20260205025010.png]]

**Pros**
- Natural parallelism
- Potentially scalable for systems with many processors

**Cons**
- Communication overhead
- Load imbalance, for some problems, it is difficult to statically partition subproblems in a balanced way
	- Dynamic load-balancing algorithms are needed
- Not easy to find the threshold (the optimal granularity that reduce overheads)
- The merging phase can be complex and challenging to do in parallel

# References