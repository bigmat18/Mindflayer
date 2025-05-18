**Data time:** 19:12 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Crossbars

Is a fully interconnected structure with $N²$ dedicated links. So **maximum bandwidth** and **minimum latency** suitable for limited parallelism only (ess. N=8) because of **link cost** and **pin-count** reasons.

![[Pasted image 20250518025038.png]]
Directly connects all N units in the blue set with all N units in the red set. Highest link cost $O(N²)$ and pin count issues **Maximum [[Processing Bandwidth|bandwidth]]** $O(N)$, **[[Communication Latency|latency]]** O(1), **node degree** N It is a non-blocking network with **diameter** 1 and **[[Bisection Width]]** $N²/2$ (with N even) For moderate values of N, it can be implemented by a single **firmware unit** (switch) with parallelism at the clock cycle level.

# References