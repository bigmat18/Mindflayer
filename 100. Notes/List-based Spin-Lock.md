**Data time:** 00:49 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# List-based Spin-Lock

We can design a list-based version of the spin-lock which is **fair** and adopts **[[RMW Instructions]]**. Every process/thread using the lock allocates a new node structure (node) having two fields:
- `must_wait` (boolean)
- `next` a pointer to the next node structure

The lock is a pointer to the last node. At the beginning, it is initialized to NULL (empty list). The implementation uses the RMW: **TSL** and **CAS**, so it is a bit more complex that the [[Array-based Spin-Lock]] implementations.

![[Pasted image 20250520005349.png | 600]]
# References