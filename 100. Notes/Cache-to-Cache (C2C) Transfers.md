**Data time:** 15:06 - 21-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Cache-to-Cache (C2C) Transfers

In **cc-[[SMP Symmetric Multi-Processor]]** or **cc-[[NUMA - Non Uniform Memory Access]]** architectures, cache-to-cache data transfers (**C2C**) are very common. The interpretation of LOADs and STOREs might cause the transmission of specific firmware messages among caches.

![[Pasted image 20250521150852.png]]

- Using the same [[Introduction to Interconnection Networks|network]] for shared memory and inter-processor communications, or **separate networks**: rend of recent CMPs
- Used for **various kinds of C2C communications** in CC protocols

C2C transfers can be intra-chip (between caches on the same CMP), or between CMPs (through the EXT-INFs)
# References