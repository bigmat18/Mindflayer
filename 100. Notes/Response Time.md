**Data time:** 16:15 - 24-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Computer Science Metrics]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Response Time

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

# References