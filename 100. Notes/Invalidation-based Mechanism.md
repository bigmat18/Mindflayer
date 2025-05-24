**Data time:** 14:59 - 21-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Invalidation-based Mechanism

In example provided in [[Cache Coherence Problem]]  $C_Q$ is no updated. PE_Q must be prevented from using the S cache line $C_Q$ until the system renders $C_Q$ constant.

The **Solution** is to force the execution of **LOAD S by Q** to generate a **miss**, so $C_Q$ will read  the updated S from M (or from $C_P$)
```
P:: { wait(go); S = F(V, S); notify(ready); }
Q:: { R = G1(W, S); notify(go); wait(ready); R = G2(R, S); }
```

![[Pasted image 20250521150347.png]]

The cache line is $C_Q$ is **invalidated** by the execution of **STORE S by P**. This instruction is **synchronous** (ie it finishes only when the invalidation has been done in $C_Q$). Alternatively, we need a **fence** before the notification. This guarantees that, when **LOAD S by Q** is executed, $C_Q$ does not contain S.

### Ping-Pong Effect
One of the problems of invalidation-based solutions in the possible of **ping-pong effect**. More PEs, which modify the same cache line simultaneously might invalidate each other repeatedly. The presence of **processor synchronizations** typically alleviates or eliminates the ping-pong effect.
###### Case 1 [[Event Notification]]
```
P:: { S = F(V, S); notify(ready); }
Q:: { wait(ready); S = G(S); }
```

It is likely that S is no longer needed by P after the notification, i.e., the invalidation from Q to P has no effect.
###### Case 2 [[Locking]]
```
P:: { lock(L); <CS1>; unlock(L); }
Q:: { lock(L); <CS2>; unlock(L); }
```
After the acquisition of the lock by P, the cache line of L is no longer needed by the same PE until the unlock.

In summary when a processor modifies a cache line, all other copies in other caches are invalidated instead of being updated. It is best for write-intense workloads and require less bandwidth but exchange more messages
# References