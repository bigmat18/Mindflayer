**Data time:** 18:33 - 21-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# MSI Snoopy Protocol

It would be useful that the same cache line can be cached by PEs, which use (read-only) their copies in parallel. So we have now three states for a given line $b$ and $PE_i$
- **Modified (M)**: the local cache of $PE_i$ is the only one having a valid copy of $b$ the memory is not updated.
- **Shared (S)**: the local cache of $PE_i$ has a copy, other caches might have the same copy valid, the memory is updated.
- **Invalid (I)**: the local cache of $PE_i$ does not have $b$

The FSM of the MSI protocol (related to a given **PE** and $b$) is the following:

![[Pasted image 20250521184017.png]]

MSI protocol has some limitations:
- Consider the case of $b$ allocated only in the cache of $PE_i$ and in the **SHARED** state
- So memory is updated too
- What happens if $PE_i$ writes the same line?
- The state becomes **MODIFIED** and the bus is accessed to notify to all PEs the write-hit activity (which is useless since no other PE has the line)
Transition from I -> M and from S -> M triggered by a local write might invalidate $b$ in other caches.

# References