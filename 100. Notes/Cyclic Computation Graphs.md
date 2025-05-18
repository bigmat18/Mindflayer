**Data time:** 02:06 - 13-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Cyclic Computation Graphs
Analysis of cyclic computation graphs modeling «client-server» computations.
### Response Time
Consider a configuration where we have a sets of **clients** $C_1, \dots, C_N$ and a **server** S (no specific semantics of client and server).

![[Pasted image 20250513175202.png]]
This model the physical model with a sigle queue where all the client send message and the server reads.

- Server [[Ideal Service Time]] is $T_{id-S}$ and [[Communication Latency|latency]] $L_S$
- [[Inter Calculation Time|inter-departure time]] from clients $\{T_{cl_i} | i = 1, \dots, N\}$
- Server [[Inter Calculation Time|inter-arrival time]] $\frac{1}{T_A} = \sum_i \frac{1}{T_{cl_i}}$
- Server [[Utilization Factor]] $\rho = \frac{T_{id-S}}{T_A}$

Semantics is **blocking after service (BAS)**. Difference between latency and **response time** ($R_Q$)

![[Pasted image 20250513175625.png | 500]]
The latency is time to procede a request from client and send back. Response time instead is bigger then latency because include in latency the average time spends from the requests in the queue. The response time is obviously related to the ideal service time of the server, the server is slow client become slow.

### Queueing models
We have a queue and a **service rate** $\mu$ (how many inputs server compute per seconds). Simple closed-form models assume **exponential inter-arrival times** with mean $\lambda^{-1} = T_{A}$. We have different options for the service time distribution:
- **Exponential Service** (M/M/1 queue) $W_Q = \frac{\rho}{\mu(1 - \rho)}$
- **General Service** (M/G/1 queue) $W_Q = \frac{\rho}{2\mu(1-\rho)}(1 + \mu² \sigma_s)$
- **Constant service** (M/D/1 queue) $W_Q = \frac{\rho}{2\mu(1-\rho)}$

![[Pasted image 20250513180629.png | 400]]

These models assume an **infinite queue**. Other models exists for queues with capacity K (ess M/M/1/K).

The response time becomes a critical performance parameter to optimize in **cyclic computation graphs** also called **[[Introduction to application layer|client-server]] computations**.

![[Pasted image 20250513181011.png | 550]]
The system of equation:
- **Fist** is the inter-departure time from the client
- **Second** is the response time of the server equal to the queue plus latency of the server
- **Third** is the utilization factor is idea service time of server divided by inter-arrivial time of server
- **Fourth** is the inter-arrival time of the server that depend to inter-departure time of client

**Self-stabilization behavior**: An increase in the inter-arrival time causes the utilization factor decrease, consequently the server response time decrease by lowering the inter-arrival time itself. ie $T_A \uparrow, \rho \downarrow , W_Q \downarrow, R_Q \downarrow, T_{cl} \downarrow, T_A \downarrow$ 

### Client-Server Parallelization
We assume the number of messages (requests) in the server's queue is limited by the number of clients. The main performance parameter is the **effective service time of the clients** which depends:
- on the clients's behavior
- mainly depends on the response time of the server

To increase the bandwidth of the clients we need to parallelize the server.

![[Pasted image 20250514005045.png | 300]]

Parallelization of the server such that:
- Its **service time** is reduced, ($\rho$ reduced) and thus also $W_Q(\rho)$
- Its **[[Communication Latency|latency]]** is reduced.

Both serber's service time and latency are directly of interest here.

###### Client-Server Example
System composed of $N\geq 1$ **identical clients** connected to a server with a request-reply pattern. Each request contains an array A of $L \geq 1$ integers. The server encapsulates an internal, already initiliazed, array B of size L. The reply consists of an array C of the same size.

![[Pasted image 20250514010251.png | 550]]

We study the system using the **cost model** (system of equatios) presented above. The goal is approximate the value of the **response time**.

Suppose now $T_{trasm} = 5\tau$. We have $L_{com} = T_{send}(L) \approx 10 L\tau$. Furthermore we assume:
- [[Ideal Service Time]] of a generic client is $T_{id-cl} = 15L\tau$
- We have 10 identical client, $N = 10$
- Server ideal service time and latency $T_{id-S} = 144L\tau, L_s = 151 L \tau$

![[Pasted image 20250514010814.png | 450]]

In this example we have $\rho \approx 0.95, W_Q \approx 1319 L\tau, R_Q \approx 1470L\tau$. The clients have very low efficiency.
$$\epsilon_{cl} = \frac{T_{id-cl}}{T_{cl}} = \frac{15L\tau}{1485L\tau} \approx 0.01$$
In this example, most of the time, the client wait the server.
#### Farm
We study the parallelization of the server and the effect on the response time and overall system’s performance (i.e., ideal service times of the identical clients). The first solution, since the computation is stateless, is to apply a [[Farm]] to replicated the server.

![[Pasted image 20250514011857.png | 500]]

The right number of worker is equal to the number of the clients, in this case 10. More number of worker is useless. Like is written in the image below:

![[Pasted image 20250514012030.png | 150]]

We obtain $\rho = 0.72, W_Q \approx 19M\tau, R_Q \approx 180M\tau$. The client efficiency is better than the privies case. But we still have $\epsilon_{cl} = 0.07$.
#### Map
The server can be parallelized using a **[[Introduction to Data Parallelism|data parallel]] approach** (that will be studied in the future lectures), where all the workers compute a partition of the same output. A different component called [[Gather]] copy the different partition of the output array and merge them in a unique array.

![[Pasted image 20250514012708.png | 500]]

The map parallelism degree is not limited by the number of clients, i.e., a minimum exists. In the example with $n_w = 29$ we obtain $\rho = 0.89, W_Q \approx 21L\tau, R_Q \approx 41L\tau$. The client efficiency is $\epsilon_{cl} = 0.27$

![[Pasted image 20250514013057.png | 150]]

We report the functions of some interesting parameters in graphic form (we use the M/M/1 model). The **qualitative shape** of the various functions can be understood by reasoning about the system behavior.

![[Pasted image 20250514012934.png | 550]]

Slower clients improve the utilization factor of the server. Both the waiting time and the response time are thereby reduced.

Let us study the [[Relative Efficiency]] of a generic client that can be conveniently rewritten as follows:

![[Pasted image 20250514013025.png | 550]]
# References