**Data time:** 19:25 - 21-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# MOESI Snoopy Protocol

In the [[MESI Snoopy Protocol]], the transition from **Modified** to **Shared** must update the copy of the line in memory. To avoid this, we can introduce a new state:
- **Owner (O)**: the line b is present in the cache $PE_i$. It is other caches in the **Shared** state and the memory is not updated.

Transitions from **Modified** to **Owner** designate the cache as the **owner**, who will write the line in memory when i is evited.

![[Pasted image 20250522004514.png]]

Protocol adopted by the **AMD Opteron** processor family.
# References