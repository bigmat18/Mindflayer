**Data time:** 14:52 - 21-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Update-based Mechanism

If we watch the example in [[Cache Coherence Problem]] we see that $C_Q$ is not correctly updated. PE_Q must be prevented from using the S cache line in $C_Q$ until the system renders $C_Q$ consistent.

The **solution** is to copy the S cache line from $C_P$ into $C_Q$
```
P:: { wait(go); S = F(V, S); notify(ready); }
Q:: { R = G1(W, S); notify(go); wait(ready); R = G2(R, S); }
```

![[Pasted image 20250521145546.png]]

The copy is caused by the execution of **STORE S by P**. Furthermore, for memory ordering reasons this instruction must be **synchronous** (or we need a **fence** before the notification). Memory ordering gurantees that **STORE S by P -> LOAD S by Q** and cache coherence guarantees that the data eventually is PE_Q's cache is updated.

# References