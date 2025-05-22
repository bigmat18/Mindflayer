**Data time:** 16:02 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory Consistency]]

**Area**: [[Master's degree]]
# Memory Models & Synchronization

### Issues with [[Event Notification]]
Image two processes P and Q executed **wait** and **notify** primitives for an event implemented by a shared Boolean flag **EVENT**. Pseudocode (**data** and **EVENT** are shared)
```
P:: { data=10; notify(EVENT); }
Q:: { wait(EVENT); y=F(data, ….); }
```
Suppose a machine with **Wark Ordering** (ie, so the worst case with al possibile reordering allowed)

![[Pasted image 20250520160632.png | 550]]

- **Problem 1**: STOREs are asynchronous and they can be implemented in an order different than the program one. Example, B can complete (and become visible to PEq) before A, and Q may execute D on the old value of data.
- **Problem 2**: if the LOAD C generates a cache miss, PEq can speculatively try to execute the subsequent instruction D and then, if **EVENT** is false when the LOAD is complete, the execution of B can be committed.

A similar problem arises when we implement event notification using **inter-processor messages** using [[Local IO]]. Pseudo-code (**data** is shared):
```
P:: { data=10; notify(EVENT); }
Q:: { wait(EVENT); y=F(data, ….); }
```
Implementation (notification implemented as a sequence of MMIO STOREs) on a machine with **Weak Ordering** (i.e., so the worst case with all possible reordering allowed).

![[Pasted image 20250520161214.png | 500]]

### Issues with [[Locking]]
In machines with relaxed memory models, [[Basic Spin-Lock|spin-lock]] might not works properly (even if we implemented them with **indivisible sequence of memory accesses**)

**Example**: shared `struct` with two integers A and B initialized to zero and protected by a lock
```
P:: { lock(L); <using struct>; unlock(L); }
Q:: { lock(L); <using struct>; unlock(L); }
```

![[Pasted image 20250520161432.png | 400]]

If Q acquires the lock after P, Q might use the old values of A and B because the STOREs done by P during CS1 might not visible to Q yet.
# References