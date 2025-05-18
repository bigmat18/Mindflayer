**Data time:** 02:39 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]] [[Computer Science Metrics]]

**Area**: [[Master's degree]]
# Maximum Link Load (MLL)

**MLL** is used to estimate the maximum bandwidth the network can support, i.e., the maximum transmission bandwidth that can be injected by every node before the network saturates.

**Saturation** means that the network cannot accept any more traffic. We need to determine the most congested link for a given traffic pattern and estimate the load on that link.

**Example** with **Random uniform traffic**.

![[Pasted image 20250518024148.png]]

- The bottleneck is the one from D to E (and back)
- Half of the traffic from A, B, C and D goes through that link
- So, MLL is equal to 2
- Network saturation at 1/2 injection bandwidth.
# References