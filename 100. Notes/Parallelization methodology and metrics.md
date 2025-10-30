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

![[Pasted image 20250511144506.png | 350]]

#### [[Ideal Service Time]]
Average time interval between the beginning of the processing of two consecutive stream items
#### [[Communication Latency]]
The time need between the send of message and the receive.
#### [[Inter Calculation Time]]
Inter-Departure and Inter-Arrival time average between difference of two consecutive send or receive in a process.
#### [[Utilization Factor]]
Describe if there is a bottleneck for the computation from inter-arrival time
#### [[Optimal Parallelism Degree]]
Describe the minimum parallelism degree to achive the perfect utilization factor (remove bottleneck).
#### [[Processing Bandwidth & Throughput]]
It is the maximum amount of data that a system can compute unit of time.  The throughput instead is the real amount of data computed per unit of time.
#### [[Completion Time]]
Time to complete an input in a full system.
#### [[Relative Efficiency]]
Ration between speedup and the number of process, describe how much is efficient the system related with the number of process used.
#### [[Scalability]]
Week scalability describe how the system has constant time with increase of process and data size. Strong scalability measure how fast the parallel version is compared to sequential. 
#### [[Roofline model]]
A model to analyze the arithmetic intensive of a system and if there is memory bottleneck.

# References