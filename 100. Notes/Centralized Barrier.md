**Data time:** 01:43 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Centralized Barrier

The **idea** is the following: we can use a single **shared counter** increased **atomically** when a new entity arrives at the barrier. When the counter is equal to the number of entities, we know that all of them have reached the barrier and can go on.

The condition to exist from the barrier is based on a **spin-loop** (busy-waiting phase), similar to [[Basic Spin-Lock]], on the shared counter.

**Correctness**: atomic RMW instructions (LOAD/STORE annotations are ok, not discussed in this part we use instead [[RMW Instructions]] **fetch-and-inc** and **fetch-and-dec**)

![[Pasted image 20250520014859.png | 300]]

The implementation above is not correct. **Problem**: if we reuse the same barrier multiple times, it does not work. How can we know that alla entities have exited from the barrier before re-entering to the next one?

We can refine the previus implementation to make the barrier correctly **re-usable**. In this second implementation we need to perform a **spin loop twice** for different purposes:
- **Fist** to ensure that all involved entities have left the previous barrier
- **Second** to ensure that alla involved entities have arrived at the current barrier.

![[Pasted image 20250520015258.png | 300]]

Is it really necessary to spin twice? This is costly in general and with high [[Communication Latency]].

### Sense Reversal
Centralized barrier with **sense reversal** avoids spinning twice. The core **idea** is: besides increasing the total number of entities by one for every entity successfully passing the barrier, the barrier can use opposite values to mark every entity state as passing or stopping.

In this implementation, `counter` controls the arrivals while `sense` the spinning conditions. In the first use of barrier, each entity checks that `sense` becomes true. In the next use, it should become false.

![[Pasted image 20250520015633.png | 400]]

- Number of operations in critical path O(N)
- Space O(1)
- Use of [[RMW Instructions]] on the same memory location (high contention)
# References