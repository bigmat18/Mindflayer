**Data time:** 00:02 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Notify-based Spin-Lock

This implementation is fair (**FIFO**) and has a moderate contention overhead because busy waiting is implemented without involving the shared memory (or caches)

![[Pasted image 20250520000319.png]]

The queue is a **FIFO buffer** of PE identifiers waiting for the lock acquisition, which is manipulated through **get** and  **put**. In the lock procedure we use an **asynchronous wait**. If the I/O message arrives before the execution of the **wait** by the process/thread, an **interrupt handler** saves the presence of the I/O messages in an **in-memory data structure** (ess. Boolean flag)
# References