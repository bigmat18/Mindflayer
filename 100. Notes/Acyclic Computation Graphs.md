**Data time:** 02:06 - 13-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Acyclic Computation Graphs

We start form a **[[Dataflow|computational graph]]** G = (V,E) where each vertex $v \in V$ is labeled with its **[[Ideal Service Time]]** $T_{id - v'}$ each edge $(a,b) \in E$ by a **probability value** (ie probability that a new input produced by vertex a is delivered to b)
###### Example
![[Pasted image 20250513125751.png | 350]]

- We want to identify **[[Bottleneck Parallelization|bottleneck]]**.
- We want to evaluate the **[[Inter Calculation Time|inter-departure time]]** from each vertex
- We want to compute the **[[Utilization Factor]]** and so **[[Relative Efficiency]]** of each vertex
- We want to computer the **[[Inter Calculation Time|inter-departure time]]** from the system $T_{\Sigma} = T_{D-S_6}$

We call the analysis **steady-state analysis** because reflects the system's behavior in the long running.
#### Backpressure
Computation graphs of interest to us are characterized by **bounded queues** implementing [[Channels in Message Passing|channels]] (FIFO buffers with a fixed capacity). Unbounded buffers are possibile, however, **memory** might be consumed without limits in the presence of bottlenecks.

We assume that messages cannot be dropped, so **backpressure** must be utilized to avoid buffer overflows.

![[Pasted image 20250513131736.png | 450]]

- In the figure above, eche edge has a **fixed capacity** of three messages
- Vertex 5 is a **bottleneck**. In the long running its input channel becomes full
- Vertex 3 will not be able to produce messages according to its speed because it is **periodically blocked** to wait for a new empty slot in the buffer implementing the **channel 3-5**
- This delay propagates up to the source (**backpressure**)
#### OR Graphs
Let S be a module, its [[Inter Calculation Time|inter-departure time]] is equal to $T_{D-S} = \max\{T_A, T_{id-S}\}$
###### Multiple-Server Theorem (OR semantics)
Let S be a module sending message to $D_1, \dots, D_n$ and let $p_i$ be the probability that a message to $D_1, \dots, D_n$ and let $p_i$ be the probability that a message is delivered to the i-th destination (with $\sum_{i=1}^n p_i = 1$The inter-arrival time to any $D_i$ is
$$T_{A-D_i} = T_{D-S}/p_i$$

![[Pasted image 20250513132926.png | 400]]

In **OR graphs**, when a vertex has more output channels, each message is delivered onto one of those channels only.
###### Multiple-Client Theorem (OR semantics)
Let $S_1 \dots, S_n$ be a set of modules sending requests to D, and suppose that D's semantics is the **[[MP IO Non-Deterministic|non-deterministic]] one** (ie D starts its processing as soon as there is a message ready to be computed in **any** of its input channels)

![[Pasted image 20250513133401.png | 300]]

The **total arrive time** to D is the sum of the individual arrival rates from each input channel
$$\lambda_i = 1/T_{D-S_i}$$
We sum the arrival rates from each individual source and then we compute the inverse to get the **aggregate inter-arrival time**. 

We can also define:
- **Input bandwidth** (ie **generation rate** of inputs) of a graph can be measured with the multiple-client theorem in case of multiple sources
- **Output bandwidth** (ie **production rate** of outputs) of a graph can be measured with the multiple-client theorem in case of multiple sinks.

![[Pasted image 20250513134514.png | 500]]

We can see that a **[[Farm]]** is an OR graph because the emitter send message to a work with a probability.
#### AND Graphs
###### Multiple-Server Theorem (AND semantics)
Let S be a module sending messages to $D_1, \dots, D_n$. We suppose that each message delivered by S (or a partition of it) is always sent to all the destinations by S. The inter-arrival time to each destinations is equal to $T_{D-S}$
###### Multiple-Client Theorem (AND semantics)
Let $S_1, \dots, S_2$ be modules sending requests to D, and suppose D computes if there is a message ready in **all** the input channels. The inter-arrival time to D is equal to
$$T_{A-D} = \max_{i=1}^n = \{T_{D-S_i}\}$$
![[Pasted image 20250513135716.png | 500]]

For example a **[[Dataflow]]** can be a AND graph if a node send messages to multiple nodes.
### Bottlenecks analysis
We will apply our parallelization methodology to **acyclic computation graphs** having the **OR semantics**. We will be able to identify **bottlenecks** and eliminate or mitigate the through **structured parallelism patterns**.

Each structured parallelization has its parametric graph (they can be **OR** or **AND** graphs depending on the paradigm)
###### Example 
![[Pasted image 20250513140224.png | 400]]

![[Pasted image 20250513140251.png | 400]]

![[Pasted image 20250513140319.png | 400]]

###### Example 
Let G be an **acyclic OR computation graph** with four nodes in the picture below:

![[Pasted image 20250513141017.png | 500]]
###### Example
Let G be an **acyclic OR computation graph** with four nodes as in the picture below:

![[Pasted image 20250513141137.png | 500]]

The source is **delayed** due to bottleneck in **S2**. We have to correct the inter-departure time from the source, like in the following way:

![[Pasted image 20250513141251.png | 300]]

Now we can continue the analysis by studying S3:

![[Pasted image 20250513141914.png | 500]]


We can define a **general method** to analysis acyclic graphs. Let G = (V,E) be an acyclic graph with **OR semantics** and with a **single source** node. Let $T = \{v_1,v_2, \dots, v_n\}$ be a **topological ordering** of G where $v_i \in V$ for $i = 1\dots n$ and $|V| = n$,

**Invariant**: when we visit $v_i$, all the nodes $v_j$ with $j < i$ have been visited and their steady-state utilization factor is $\rho_j \leq 1$

When we visit $v_i$ we know $T_{D-v_k}$ for all nodes $v_k \in N_{IN}(v_i)$ so we can compute $T_{A-v_i}$ using multiple-client theorem. We have two possibilities:
- **Non-bottleneck**: $\rho_{v_i} \leq 1$ so $T_{D-v_i} = T_{A-v_i}$ we can proceeded with node $v_{i+1}$ (note that the invariant is preserved) 
- **Bottleneck**: $p_{v_i} > 1$, so $T_{D-v_i} = T_{id - v_i}$. We have to correct the speed of the source (its effective inter-departure time) and restart the analysis from the beginning.
We have to find a way to correct the source's speed and to preserve the invariant. 

**Flow conservation**: at the steady state, the inter-arrival time of a module matches its inter-departure time.

Consider the case we reach $v_i$ and it is a bottleneck. Since all proceding vertices have utilization factor less or equal than 1 (by invariant), we can write the inter-arrival time to $v_i$ as a function of the actual $T_{D-v_1}$

![[Pasted image 20250513143355.png | 450]]

The blue condition derives from **flow conservation** (ie at the steady-state the inter-arrival time to a vertex must be equal to its inter-departure time). Let $T'_{D-v_1} = T_{D-v_1} \cdot \alpha$
$$\alpha = \frac{T_{id-v_i} \cdot P(v_1 \to v_2)}{T_{D-v_1}} = \frac{T_{id - v_1}}{\frac{T_{D-v_1}}{P(v_1 \to v_2)}} = \rho_{v_1}$$
###### Example 
Example of the algorithmic approach. We start from the graph bellow:

![[Pasted image 20250513144902.png | 350]]

- **Step 1**: We start from the source and we visit the nodes until we find the fist bottleneck (node S5)
![[Pasted image 20250513145010.png | 400]]

- **Step 2**: we restart from the source with its new inter-departure time, and we look for the next bottleneck (it is in node S7)
![[Pasted image 20250513145114.png | 400]]

- **Step 3**: we complete the analysis by restarting from the source to re-compute the right inter-departure times
![[Pasted image 20250513145134.png | 400]]

##### Steady-State analysis
If we arrive at a **bottleneck** we correct the source's inter-departure time by multiplying it by the [[Utilization Factor]] of the bottleneck ($\alpha$). Then we restart the analysis. The **invariant** is preserved since when we reach again $v_i$ we have $p_{v_i} = 1$. At the worst case we restart the analysis every time we visit a new node.

![[Pasted image 20250513145908.png | 250]]

The **worst-case complexity** $O(|V|\cdot|E|)$, which is $O(|V|³)$ for dense graphs.

### Bottleneck-Elimination Analysis
In many cases instead, we need to **eliminate bottlenecks**. Now we see how we can do it with an example.

- **Step 1**: we start the analysis from the Source and we visit the first vertex S2
![[Pasted image 20250513150228.png | 400]]

- **Step 2**: we paralleize S2 (suppose we can apply [[Farm]]). We eliminate completely the bottleneck
![[Pasted image 20250513150857.png | 400]]

- **Step 3**: we can visit the next node of the graph S3 and discover whether it is a bottleneck or not
![[Pasted image 20250513150933.png | 400]]

- **Step 4**: suppose S3 is suitable to be parallelized with a [[Pipeline]] (partitioned r/w state). However, stages are not balanced.
![[Pasted image 20250513151029.png | 400]]

- **Step 5**: the bottleneck in S3 has not been eliminated completely. We have to restart the analysis from the Source
![[Pasted image 20250513151147.png | 400]]

- **Step 6**: we can reduce the parallelism applied by the farm in S2 to save efficiency. Then, we visit S4.
![[Pasted image 20250513151227.png | 400]]

Bottleneck elimination (**B-Elimination**) is similar to the steady-stare analysis. However, if we find bottleneck, we try to parallelize it. If the parallelization eliminates the bottleneck, we proceed to visit the next node. Overwise, we restart the analysis from the source to **correct** the **[[Optimal Parallelism Degree]]** of the already parallelized vertices to save **efficiency**.

![[Pasted image 20250513151452.png |250]]

In the **worst-case complexity** we have $O(|V|³)$ as before.
# References