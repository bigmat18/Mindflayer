**Data time:** 00:41 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Array-based Spin-Lock

The main question that this implementation wont resolve is why not giving to each processor a **private variable** to implement busy waiting, because in that case traffic in the network becomes **O(1)**.

**Memory Layout**: an array names `slots` of $N>0$ integers. Two possibile value:
- `go_ahed` macro equal to 0
- `must_wait` macro equal to 1,
We also keep a counter called `next_slot`

**Initialization**: first element of `slots` initialized to `go_ahead` the reset to `must_wait`. The counter `next_slot` is set to zero.

![[Pasted image 20250520004721.png | 500]]

- **PROS**: spinning on local variables only. Fair implementation. 1 RMW per processor.
- **CONS**: higher uncontended overhead than [[TS-based Spin-Lock]]. Higher storage (linear in N): 128 processors, byte padding (to avoid **false sharing**)

# References