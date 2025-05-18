**Data time:** 20:23 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# CLOS Networks

[[Indirect Networks]] with three stages: 
- **input stage** is used to connect source nodes
- **output stage** to connect destination nodes
- **middle stage** is between them

A CLOS network is identified by three parameters $(m,n,r)$. The network interconnects $N = r \cdot n$ nodes in the **lest set** with other N nodes in the **right set**

![[Pasted image 20250518202951.png]]
- **Maximum Bandwidth** O(N)
- **Latency** O(1)
- **Node degree** $\max\{n + m, 2r\}$
- **Diameter** is 3
- **Bisection width** is $r \cdot m$ (with r,m even)

**Theorem (1953)**: CLOS are **non-blocking** if $m \geq 2n - 1$
This because when setting up a new flow, the source switch will at most have n-1 already existing flows, which are routed to n-1 middle switches. Similarly, the destination switch will at most have n-1 existing flows. Hence, existing flows will at most use 2n-2 middle switches.

![[Pasted image 20250518203310.png | 500]]

If $B\to H$ and $D\to E, A$ cannot communicate with F. However, if $m \geq n$ CLOS networks are **rearrangeable non-blocking**

![[Pasted image 20250518203430.png | 500]]

### Hierarchical CLOS
CLOS networks provide **path diversity**. From any source unit to any destination unit there exists one distinct path for each middle switch (i.e., **m distinct paths**). **Modular composition** of CLOS networks: we can increase path diversity by replacing each middle switch with a CLOS network.

![[Pasted image 20250518203831.png | 600]]

CLOS network (**left**) is **rearrangeable non-blocking**. Also the one on the right-hand side is rearrangeable non-blocking because (2,2,2) CLOS is.

###### Example
Myrinet-2000 is a CLOS network for 128 hosts.

![[Pasted image 20250518205039.png | 400]]
# References