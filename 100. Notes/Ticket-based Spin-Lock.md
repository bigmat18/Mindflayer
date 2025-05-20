**Data time:** 00:21 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Ticket-based Spin-Lock

This type of spin-lock there is because [[TS-based Spin-Lock|TTS]] has two major flaws:
- it is **unfair** and might generate starvation
- It generate a lot of **traffic** when the lock is released and manu threads/processes are waiting for the lock acquisition

**Ticket spin-lock** solves some of these issues. It use two integers: `next_ticket` and `serving_ticket` initialized to zero. It is **fair**, it requires a single [[RMW Instructions]] to acquire the lock and a single STORE to release it.

![[Pasted image 20250520002722.png]]

**FAI Rval, Rlock, \#value** Load memory location with the address in Rlock into Rval. Then atomically increases it by value and write the result in the same location.

- **PROS**: only one RMW instruction per processor. Probing with normal LOAD only. FIFO scheme
- **CONS**: a lot of traffic in the network during polling on the same variable `serving_tickety`

# References