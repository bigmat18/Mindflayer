**Data time:** 19:47 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Omega Networks

**Omega Networks** connect N nodes in the **left set** with N nodes in the **right set**. The number of stages is $\log_2 N$. At each stage, i-th input is connected to the j-the output such that
$$j = \begin{cases}2i & 0\leq i \leq N/2\\ 2i + 1 - N & N/2 \leq i \leq N\end{cases}$$
This communication pattern is also called **perfect shuffling**. The figure below show it with N=8

![[Pasted image 20250518201045.png | 500]]

Example of an **Omega Network** having N=8 endpoints in the two sets respectively. The number of switches is $N/2 \cdot \log_2(N)$. Omega networks **repeat** the same **perfect shuffling pattern** at each stage (except for connecting the last stage with endpoints in the **right set**)

![[Pasted image 20250518201452.png]]

- Similar structure to 2-ary n-fly networks (butterflies)
- However, we have a different connectivity rule
- Still exactly one path for each pair (src, dest)
- It is a  **blocking networks** having node degree 4
- **Maximum [[Processing Bandwidth|bandwidth]]** is O(n)
- **[[Communication Latency|Latency]]** $O(\log_2 N)$ and **diameter** $\log_2N$
- **[[Bisection Width]]** is N/2

### Deterministic Routing
Omega networks do not provide path diversity (analogously to [[Butterfly (2-fly n-ary) Networks]] and k-ary n-fly in general). Since only one path exists between any pair of source and destination nodes, the routing algorithms is **deterministic**. A very easy **left-to-right algorithm** considers the binary representation of the destination identifier only.

![[Pasted image 20250518202322.png]]

# References