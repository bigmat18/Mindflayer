**Data time:** 14:14 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]]

**Area**: [[Master's degree]]
# Parallelization methodology and metrics

There are a large set of definitions and terminology very useful in high performance computing and parallel systems topics. Let us consider an example of parallel computation developed using **message-passing model**

![[Pasted image 20250511143909.png]]

We have **two process** with private addressing space and a channel:
- **P (producer)** produce a stream.
- **Q (consumer)** consume the stream of P.
- **input_stream**: channel that connect the two process

If P is very slow also Q will be blocked to the first instructions on the while loop. We would like to know the **calculation time** of the process Q that is the time spent by Q in computing one generic input x.
```
NEW: <runtime support of receive(x)> 
//Implementation of receive, and the of this fun in Rx-addr we have the results 

	LOAD Rx-addr, #0, Rx
	CLEAR Ry
	CLEAR Ri

// In RA-base there is the initial address of A
LOOP: LOAD RA-base, Ri, Ritem
	IF != Rx, Ritem CONT
	INCR Ry // 1/2 probability
CONT: INCR Ri

// In RL the foxed size of array
	IF < Ri, RL LOOP
	STORE Ry-addr, 0, Ry
	<runtime support of send(y)>
	GOTO NEW
```

Processing time is **O(L)** dominated by the loop instructions (4.5 instructions per iterations on average, probability to find an occurrence of x, 4 sure instruction + 1 with 1/2 probability the INCR Ry)

![[Pasted image 20250511144506.png ]]

#### [[Ideal Service Time]]
#### [[Inter Time]]
#### [[Utilization Factor]]
#### [[Optimal Parallelism Degree]]

### Bottleneck Parallelization
For first we have identified the **bottleneck condition** ($\rho > 1$) and next design a proper parallelization. The **goal** in a **stream-based scenarios** the parallelization goal is to reduce the ideal-service time of the parallelized module s.t. it matches the inter-arrival time, i.e.
$$T_{id-Q}(n) = T_{A-Q}$$
$T_{id-Q}(n)$ is the [[Ideal Service Time|ideal service time]] with a parallelism of n. The ideal case (without performance degradentions) the [[Optimal Parallelism Degree|right parallelism degree]] use the optimal one:
$$T_{id-Q}(n) = \frac{T_{id-Q}(1)}{n} \:\:\:\:\:N_{opt} = \bigg\lceil\frac{T_{id-Q}(1)}{T_A}\bigg\rceil$$
In case performance degradation exist (e.g., load imbalance), the optimal parallelism degree might be insufficient, or no parallelism degree can remove the bottleneck with a given parallelization. In all cases, the chosen parallelization strategy shall preserve the **computation semantics**.

In the example above the computations is **[[HTTP|stateless]]** there is side effects in the function. Therefore, functional replication is a feasible parallelization strategy for the problem.

![[Pasted image 20250511161859.png]]

We have 4 replica of the same function Q. When an input arrive it goes to one of free replica. If all replica are busy the input wait to be processes. In this cases **we don't change [[Networks Metrics|latency]]** 
#### [[Processing Bandwidth]]
#### [[Completion Time]]
#### [[Communication Latency]]
#### [[Relative Efficiency]]
#### [[Scalability]]
#### [[Roofline model]]

# References