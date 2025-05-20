**Data time:** 00:15 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# TS-based Spin-Lock

Instead of LOAD/STORE with annotations, we can develop a spin-lock using the **[[RMW Instructions|Test&Set]]** atomic instruction. We can assume the presence of the instruction: 

```
TSL Rx, Ry, #value
```

**Semantics**: read in Rx the value of the memory location whose address is in Ry, then atomically write in the same location the constant value

Below, the implementation in pseudo-code and D-RISC:

![[Pasted image 20250520001744.png | 500]]

### TS Spin-Lock Performance

![[Pasted image 20250520001843.png | 400]]

### TTS Spin-Lock
The TS spin-lock repeats an atomic sequence (one LOAD and one STORE to read and to write the lock flag). We can design the spin-lock to **spin** (most of the time) on a **local variable** (this is done efficiently in the local cache). We call this version **test-test-and-set spin-lock (TTS)**.

Slightly higher latency than TS spin-lock in uncontended case. Still **unfair** but generates less traffic (**more scalable**) and the same memory requirements (one flag only).

![[Pasted image 20250520002145.png]]
# References