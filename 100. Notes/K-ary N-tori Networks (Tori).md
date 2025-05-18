**Data time:** 22:31 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# K-ary N-tori Networks (Tori)

This type of network is a **Direct Network** also called **Tori**. Like [[K-ary N-mesh Networks (Meshes)]] but with **wrap-around links** between edge nodes (not good for a planar surface such as on-chip networks). They are **edge symmetric** (better for load balancing because on meshes traffic concentrates in the center of the network). Better to be used in 3D spaces (supercomputers). They are still **blocking networks**

![[Pasted image 20250518224047.png | 500]]

**Low-dimensional meshes/toris** have higher theoretical latency but are composed by less complex switches.

### MLL
Let's study the [[Maximum Link Load (MLL)]] of a toroidal grid (4-ary 2-tori). Divide the network in half (ess. eight nodes on the left and other eight nodes in the right-hand side of the figure below).

![[Pasted image 20250518224244.png | 500]]

What happens in case of 4-ary 2-mesh (so without toroidal links)? Maximum Link Load increases to 1 ([[Bisection Width]] becomes 4).

### Deterministic Routing
A simple routing approach for **k-ary 2-mesh networks** exploits one of the possible **shortest paths**. Source and destination identifiers are expressed as a pair of **binary coordinates**, ess $(X_s, Y_s)$ and $(X_D, Y_D)$

**Algorithm**: the first dimension is followed until we reach the switch with coordinates  $(X_D, Y_S)$. Then, we follow the second dimension reaching the switch connected to the destination.

![[Pasted image 20250518224924.png | 300]]

This algorithm is trivial and far from being efficient. **Path diversity** can be exploited to derive a**daptive routing algorithms**. Avoiding passing through the center of the network to alleviate network conflicts.

# References