**Data time:** 01:00 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Notification with Shared Flags

The notification mechanism can be implemented:
- using **shared variables** (without atomic sequences of memory access)
- using **inter-processor communications** via **[[Local IO]]**. This solution need a fixed **pinning** of process/threads onto PE

The signature of the two primitives is the following
- `notify(event_identifier)`: set the presence of the event
- `wait(event_identifier)`: put the calling processor in a waiting state (busy waiting) until the event is present.

As said, out implementations will assume exactly one **notify** executed before a **wait** on the same event (otherwise?)

![[Pasted image 20250520010629.png | 500]]
# References