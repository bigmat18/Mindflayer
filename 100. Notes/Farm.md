**Data time:** 13:47 - 12-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]] [[Parallel and distributed systems. Paradigms and models]]

**Area**: [[Master's degree]]
# Farm
**Farm** is a stram-parallel paradigm based on the replication of a **purely functional computation** sometime it is called also **master-worker** (computation without state, **stateless**).
- **Objective**: **improving the [[Processing Bandwidth & Throughput|thoughput]]** of a single stage
###### Example
A process P receives a stream of images and applies on each image a filtering procedure (we assume it is stateless since the output image depends solely on the value of pixels of the corresponding input image)

![[Pasted image 20250512155604.png]]
- **Emitter (E)** is in charge of distributing inputs to workers
- **Workers (W)** are identical copies of the original P (black-box)
- **Collector (C)** is in charge of collecting results and multiplexing them in a single physical output stream
- **Master (M)** implements both **Emitter** and **Collector** functionalities.
### Input-Output Distribution
The **[[Optimal Parallelism Degree|parallelism degree]]** is the number of workers $N > 0$ (it must be equal to the optimal parallelism degree based on the ideal service time of P an the [[Inter Calculation Time|inter-arrival time]])

Emitter should **distribute** the **load** evenly among workers. So, the **distribution strategy** is very important:
##### Round-Robin strategy (RR)
Inputs i goes worker $j = i \mod N$. RR strategy provides the same number of inputs to each workers. This grantees **load balancing** provided that **calculation time** per input is **constant**. Overwise we can have problem like in the example below: 

![[Pasted image 20250512160746.png]]
Where task 4 should be distributed to worker 2 that is idle, but is forced to go to worker 1 (busy).
##### On-demand strategy (OD)
Each input is distributed dynamically to a worker that is ready to compute it. For example:

![[Pasted image 20250512161218.png]]

Where after we sent task 3 to worker 3 we don't send anything else to it because it is busy instead we continue to send task to worker 1 and 2 because they had finish their task.

The implementations can be **deterministic** or **[[MP IO Non-Deterministic|non-deterministic]]**:

![[Pasted image 20250512161448.png]]

- **Deterministic distribution** and **collection strategies** are capable of preventing **output disordering** at the expense of a potential **load imbalance**
- **Non-deterministic distribution** and **collection strategies** require proper solutions to **disordering**

![[Pasted image 20250512162642.png]]
###### Solution 1
Inputs are emitted with **unique identifiers** that will be propagated with the corresponding outputs. Sorting them is responsibility of the destination process/system using the outputs computed by the farm.
###### Solution 2
Inputs are emitted with **unique identifiers** that will be propagated with the corresponding outputs. Sorting them is responsibility of the Collector itself that will queue pending results that cannot be produced immediately (otherwise the ordering is not preserved).


The Emitter can be implemented as a passive entity (one or more task queues), the queues are lock-protected or implemented in a lock-free way:
- **Work-Sharing:** all Workers share a single concurrent task queue
- **Work-Stealing**: each Worker maintains its local task queue; tasks may be assigned to Workers using a RR or random policy; idle Workers (i.e., those with empty local queue) try to steal (a batch of) tasks from other Workers.

![[Pasted image 20260205020448.png]]
### Cost Model
From the performance viewpoint, a farm can be studied as a **pipeline of three stages** (two sequential, one parallel stage).

![[Pasted image 20250512164438.png | 550]]

The **[[Inter Calculation Time|inter-departure time]]** from the farm is
$$T_{\Sigma} = \max\{T_A, T_{id-\Sigma}\}$$
To remove the [[Bottleneck Parallelization|bottleneck]] provided that load is balanced, we need to use $N_{opt} = \lceil T_{id-W} / T_A \rceil$. Ideally, the [[Processing Bandwidth & Throughput]] increases proportionally with the number of workers.

### Finding the number of workers
Let $T_a$ be the arrival time of stream items (tasks) to a farm computing the function F with k Workers. To avoid the stage of Workers being a bottleneck, it must hold:
$$
\max\{T_s^{e}, T_s^{c}\} \leq \frac{T_s^{w}(F)}{k} \leq T_a
$$
therefore
$$
\frac{T_s^{w}(F)}{T_a} \leq k \leq \frac{T_s^{w}(F)}{\max\{T_s^{e}, T_s^{c}\}}
$$
the minimal $k$ that satisfies both inequalities is:
$$
k_{opt} = \bigg \lceil \frac{T_s^{w}(F)}{T_a}\bigg \rceil
$$
### Stateful Workers
Traditionally, farm patterns assume stateless Workers: each task is processed independently, and no internal data persists across tasks. However, some applications require stateful computations 

##### Read-Only State
Farm is feasible for stateless computations. No worries if a state exists and it is **read only** ie we can replica it. **State replication** if workers are processes with independent addressing spaces (higher memory occupancy).
- **Distributed Memory:** each Worker holds a replica of the read-only data. There is no need for synchronization beyond the initial data distribution
- **Shared Memory**: A single shared copy of the state is sufficient


![[Pasted image 20250512165531.png | 600]]

 Each worker counts the number of occurrences of the received inputs by **reading** its copy of the whole array A.
##### Read-Write State
If the state is **modifiable**, replicating it might produce wrong result. However, if **additional properties** on the computation are identified, we can derive some "special utilization" of farm. For example:

![[Pasted image 20250512171203.png]]

- **Monolitich (non-partitionable) state**: the entire data must be updated whenever a write occurs
	- Workers share or replicate a single logical state
	- The performance depends on the probability of updates w.r.t. reads operations. T**he cost scales with the frequency of writes.**
	- **The multiple-reader single-writer locking** schema can be used on shared-memory systems
	- **Multicast-based approach** in distributed-memory systems: tasks that update the state are broadcasted to all Workers to update their local state (or, equivalently, the updated state is broadcasted to all Workers) 
- **Partitionable state**: divide the global state into disjoint segments so each worker handles a portion
	- Apply some form of hashing to the tasks that identifies the partition and send the task to the proper Worker
	- Workers are not anonymous. The Emitter must send a given task to one (or a subset of) Worker(s)

![[Pasted image 20260205021425.png]]

### Key-Partitioned Farm
Another relevant case of farm with modifiable state enforces another property of the state and of the input stream. The input stream conveys inputs belonging to a fine set of classes identified by a **key attribute**. We assume $K>0$ distinct keys from 0 to $K-1$

The **state is partitioned** with one partition per key (**partitioned stateful**).
![[Pasted image 20250512171536.png]]

Let $p_i$ be the probability to receive an input with i-th key. We have:
$$\sum_{i=0}^{K-1}p_i = 1$$
Let $H_i$ be the set of keys assigned to the i-th worker. So:
$$\bigcup_{i=0}^{n-1} H_i = \{0, \dots, K - 1\}$$
The [[Inter Calculation Time|inter-arrival time]] to the i-th worker can be computed as 
$$T_{A-i} = T_A / \sum_{j\in H_i}p_j$$
where $T_A$ is the inter-arrival time to the emitter.

Based on the discrete probability distribution, some workers might receive much more inputs that others (**load imbalance**). Such load imbalance is not necessarily a performance issue provided that $p_i = T_{id-W}/T_{A-i} \leq 1$ for all workers.

### Specialized Workers 
Key-partitioned farm, writed above, is a special stateful case of a **farm with specialized workers**. This paradigm can be applied even when the computation on all inputs is partitioned stateful. Instead of having workers capable of processing any input, each worker can process only inputs of certain type(s). 

![[Pasted image 20250512172837.png]]

It is useful in the following situations:
- **Execution stage** of pipelines process
- General-purpose anonymous **functional units (FUs)** are expensive in terms of chip occupation
- Different operations require **different circuits** with different chip area occupation
- Again, **load imbalance** does not necessarily generate a [[Bottleneck Parallelization|bottleneck]] if each FU has [[Utilization Factor]] less then 1.
# References