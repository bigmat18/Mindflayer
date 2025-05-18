**Data time:** 23:09 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Circuit Switching

It is the strategy with coarsest granularity. The network pre-allocates resources across multiple hopes between the source and the destination. **Probe messages** are sent to reserve resources. Once probed the path, no per-hop routing decision must be made.

No distinction of the message in packets. Data flows from source to destination until a **closing message** is transmitted.

![[Pasted image 20250518231032.png]]
##### Cons
- **throughput** can suffer due to **setup** and **hold time** for establishing a path
- links are idle until setup is complete
- links cannot be shared by multiple communications even if one is currently idle
##### Pros
- good for transferring large amounts of data
- No per-packet header is needed
- no buffering capability
# References