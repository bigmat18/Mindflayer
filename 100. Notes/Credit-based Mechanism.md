**Data time:** 12:36 - 19-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Credit-based Mechanism

An alternative mechanism to [[RDY-ACK Transmission]] and [[Stop-Go Transmission]] is **Credit-based Mechanism**. The receiver advertises an initial number of flits (**credits**) that can be safely buffered. Every time the sender sends one flit, it decreases the credit counter. If it becomes zero, it stops transmitting. The receiver periodically sends credits back to the sender.

![[Pasted image 20250519125137.png | 400]]
###### Cons
- More traffic in the upstream signal
###### Pros
- Smaller buffering capacity in the receiver
# References