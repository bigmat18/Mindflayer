**Data time:** 18:27 - 21-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Minimalistic Snoopy Protocol

Assume **write-back caches** and consider below a minimalistic version of snoopy-based protocol. We have only two states of a given cache line $b$ and $PE_i$:
- **Modified (M)**: the local cache of $PE_i$ is the only one having a valid copy of $b$, the memory is not necessarily updated.
- **Invalid (I)**: the local cache of $PE_i$ does not have $b$

Some constraints in the configuration of states in caches exits, ie, a cache line can be modified (M) in one cache only. We show **Finite State Machine (FSN)** with the states and transitions between states in response to **local actions** and **remote events** snooped through the bus by a given PE.

![[Pasted image 20250521183210.png]]

Each cache line can be valid in at most one cache. So, a **LOAD with miss** will invalidate all other copies (if any). The same happens for a **STORE with miss**. So, transitions between states $I\to M$ might invalidate b in the other caches. 
# References