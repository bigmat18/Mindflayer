**Data time:** 16:16 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory Consistency]]

**Area**: [[Master's degree]]
# Safety Nets (fences)

Intel machines adopt a model quite similar to [[Relaxed Memory Models|TSO]]. ARM and Power (IBM) machines adopt relaxed memory models. The **general idea** is: most of the memory accesses, since they likely happen for **private data**, can be safely reordered by the machine to hide latencies. When specific ordering ordering requirements are needed, some **safety nets** should be used. 

A nice analogy with "balloon twisting". GAS particles (LOADs/STOREs) in the balloon can freely move (reordering). Twisting allows separating two regions (introduce a partial ordering).

Which safety nets? It depends of the machine's instruction set and the memory model implemented. They can be:
- **Memory instructions** with **special annotations**
- **Special instructions** (ie, **[[Barriers|memory barriers]]** also called **FENCE** instructions)

### Safety Nets: Notification
To implement [[Notification|event notification]] in machines with [[Relaxed Memory Models]], we first have to guarantee STORE atomicity. Second, the LOADs accessing the data by the notified process cannot precede the LOAD for reading the event, and the conditional branch to test it.

![[Pasted image 20250520164952.png | 550]]

We need to introduce a special STORE with ‘**synch**’ annotation, which has **synchronous semantics** (i.e., PE executes the next instruction when such a STORE has been globally executed). LOAD (e.g., causing a cache miss) might be reordered with next LOADs/STOREs. The annotation ‘**in_order**’ prevents this.
### Safety Nets: Locking
[[Locking]] need to be correctly implemented in machines with [[Relaxed Memory Models]]. For example, before the **unlock** we should guarantee that all the pending STORE instructions (if any) performed on shared data within the critical section are made visible to all PEs

![[Pasted image 20250520183528.png | 550]]

A **SFENCS** has to be added to the **unlock** procedure. No FENCE is required in the **lock** procedure. Indeed, **[[RMW Instructions]]** (like the TSL) implicitly introduce a [[Barriers|memroy barrier]].
# References