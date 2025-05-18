**Data time:** 19:09 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Butterfly (2-fly n-ary) Networks

Butterfly networks are a popular example of a **limited-degree indirect network** used in parallel architectures. They are built through the **modular interconnection** of a **small switched**, as exemplified in the following figure.

![[Pasted image 20250518191048.png | 400]]

**Butterfly** with **dimension** $n = \log_2 N$ (in the example above 2). 

Butterflies connect N nodes in the **left set** with N nodes in the **right set**. They are formally called **2-fly n-ary networks**.

![[Pasted image 20250518192110.png | 500]]

### Deterministic Routing
Routing (left to right) executed in n steps. At step i the switch compares the **i-th bit** of the **sender** and **destination binary representations** (bit numbering start from the most significant bit). If equal, the **straight link** is followed. Otherwise the switch routes the message onto the **oblique link**.

![[Pasted image 20250518192720.png | 300]]

**Reverse path** (right-to-left): we start from the second least significant bit. The least significant bit of the destination ID is used to choose the last link.

### MLL in Butterflies
Butterflies have **[[Maximum Link Load (MLL)]]** equal to 1. Not formal proof here, just take the example below with a 2-ary 3-fly network.

![[Pasted image 20250518193346.png | 300]]
Divide the network in two parts of the same size. The **[[Bisection Width]] = 4**, that means 4 nodes (top half) send 1/2 traffic to the lower half 4/2 = 2. This load is distributed across the channels **MLL = 1**

This is valid under **uniform traffic**. Not true for **adversarial traffic**. Example: all traffic from the top half sent to the bottom half. MLL becomes 2.

### Benes Networks
Two copies of butterfly networks $2N \cdot \log_2 N$ switches. Enhance Butterfly networks by improving **path diversity**. Furthermore, they are rearrangeable non-blocking. **Rearrangeable non-blocking** means that by rearranging the flows, all permutations can be realized without contention.

![[Pasted image 20250518194159.png | 400]]

### K-ary N-fly Networks
Butterflies can be generalized with any ariety K>1, though it must be ‘low’ for limited-degree networks.

![[Pasted image 20250518194324.png]]    ![[Pasted image 20250518194338.png]]

Network characteristics:
- Number of processing nodes $2N$ with $N=k^n$
- Number of links and switches $O(N\log_k N)$
- **Maximum bandwidth** $O(N)$
- **Latency** with $\infty$ distance $O(\log_k N)$
- **Node degree** $2k$
- **Blocking network**
- **Bisection width** N/k (with N even)
- Switch complexity for maximum bandwidth = $O(c^k)$, once the elementary crossbar is available

# References