**Data time:** 23:35 - 19-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Locking

A shared data structure (L), called **lock semaphore** is associated with a **shared object** (S). **Spin-lock** is a kind of lock where waiting phases are implemented through **busy-waiting**.

![[Pasted image 20250519233658.png | 500]]

**Example** of lock utilization with different critical sections working on the same shared data structure S.
```
lock_t *L; *L = ‘green’; // initialization //
P:: { c1; lock(L); S = F(S, …); unlock(L); c2; }
Q:: { d1; lock(L); S = G(S, …); unlock(L); d2; }
```

### [[Basic Spin-Lock]]

### [[Notify-based Spin-Lock]]

### [[RMW Instructions]]

### [[TS-based Spin-Lock]]

### [[Ticket-based Spin-Lock]]

### [[Array-based Spin-Lock]]

### [[List-based Spin-Lock]]
# References