**Data time:** 18:51 - 21-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# MESI Snoopy Protocol

**MESI** is the basic protocol adopted by **Intel Architectures**. We have four possibile states for a given block $b$ and $PE_i$:
- **Modified (M)**: the line b is in the cache of $PE_i$ only, while the copy in memory is not updated
- **Exclusive (E)**: the line $b$ is in the cache of $PE_i$ only, and the copy in memory is updated.
- **Shared (S)**: $b$ is in **more caches** including the one of $PE_i$, and the copy in memory is updated
- **Invalid (I)**: $b$ is not valid in the cache of $PE_i$

FSM of the MESI protocol (related to a given **PE** and block $b$)

![[Pasted image 20250521185608.png]]

Transitions from **EXCLUSIVE** to **MODIFIED** do no requires access to the bus. Transitions from S -> M and from I -> M invalidate the line b in other cases. **MESI** protocols has some limitations:
- Complexity of the mechanism that determines exclusiveness
- Flooding of replies in case of a line in **SHARED** state in more caches (like the example below)
- Transitions from M-> S requires to write-back the cache line in memory which is a waste of bandwidth.

##### Example 1
Consider a scenario where **four** PEs execute **LOAD** and **STORE** instructions manipulating a shared variable S stored in one cache line.

![[Pasted image 20250521190501.png | 400]]

The state of S in the LSK of a PE can change:
1. due to local PE activities
2. due to activities from other PEs (snooped through the Snoopy Bus)
At the beginning, the block od S is **INVALID** in all caches.

In case of a **LOAD with hit** the state of the cache line in the LSK does not change. No action in the Snoopy Bus. Suppose instead of **LOAD with miss** executed by **PE1**. The PE requests access to the bus. No cache has the line, so the request must be directed to the memory.

![[Pasted image 20250521191456.png | 400]]

**LOAD with miss** executed by PE2. The PE requests access the bus. The cache of PE1 knows that it has the line, and replies by sending the line to the cache of PE2. The caches of PE1 and PE2 update their LSK by changing the state of the line to **SHARED** (also the memory is updated)

![[Pasted image 20250521191547.png | 400]]

![[Pasted image 20250521191910.png | 400]]

**STORE** executed by **PE2**. **PE2** requests access to the bus and broadcasts the write activity request. All the caches having the line invalidate it by updating their LSK. The state of the cache line in the LSK of the requesting processor **PE2** is updated to **MODIFIED**.

![[Pasted image 20250521192025.png | 400]]

![[Pasted image 20250521192058.png | 400]]

**LOAD with miss** executed by **PE1**. The PE requests access to the bus. The cache of **PE2** knows that it has the line, and replies by sending the line to the cache of **PE1**. The cache line is allocated in PE1 and the state of both the LSKs is updated to **SHARED**. The memory is updated too.

![[Pasted image 20250521192239.png | 400]]

![[Pasted image 20250521192301.png | 400]]

**LOAD with miss** executed by **PE3**. The PE requests access to the bus. The caches of **PE1** and **PE2** know that they have the line, and reply by sending it to the cache of **PE3**. All the caches having a copy of S will send their reply to the PE3’s cache. This can be critical because there might be a “**flood**” of messages directed to the same cache.

![[Pasted image 20250521192429.png | 400]]

![[Pasted image 20250521192458.png | 400]]

A Variant called **MESIF** introduces a further **Forward** state of the PE in charge of sending the cache line after a LOAD with miss. This means that when PE3 execute a LOAD received only from the forward processor the block to avoid flood.
# References