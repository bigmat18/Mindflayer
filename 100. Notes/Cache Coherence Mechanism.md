**Data time:** 14:51 - 21-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Cache Coherence Mechanism

### [[Update-based Mechanism]]

### [[Invalidation-based Mechanism]]

### [[Cache-to-Cache (C2C) Transfers]]

### Invalidation vs Update
We have the following **Scenario**: $PE_i, PE_{j1}, \dots, PE_{jk}$ contain the same valid copy of a cache line in their respective caches, and $PE_i$ modifies such a line. The $PE_i, PE_{j1}, \dots, PE_{jk}$ read the it again.
##### [[Invalidation-based Mechanism|Invalid-based solution]]
Let us try to understand the firmware messages exchanged between caches 

- $PE_i$ sends (**[[Multicast]]**) on **invalidation message** to $PE_i, PE_{j1}, \dots, PE_{jk}$
![[Pasted image 20250521152452.png | 400]]
- $PE_i, PE_{j1}, \dots, PE_{jk}$ independently (in parallel) reply with a **done** message (just the header, a sort of ACK) to $PE_i$. This is not blocked, we can do other things meanwhile the ack did't arrive
![[Pasted image 20250521152833.png|400]]

- $PE_i, PE_{j1}, \dots, PE_{jk}$ independently send a **[[Cache-to-Cache (C2C) Transfers|C2C]] read request** to $PE_i$ and receive the correct value of the cache line from $PE_i$
![[Pasted image 20250521153002.png | 400]]

##### [[Update-based Mechanism|Update-based solution]]
Let us try to understand the firmware messages exchanged between caches.

- $PE_i$ sends (**[[Multicast]]**) an **updated message** to $PE_i, PE_{j1}, \dots, PE_{jk}$
![[Pasted image 20250521153423.png | 400]]

- $PE_i, PE_{j1}, \dots, PE_{jk}$ reply with a **done** message to $PE_i$
![[Pasted image 20250521153452.png | 400]]

The whole number of exchanged words is greater with invalidation. It is batter in term of number of message. However, in the update-based solution, the required **[[Processing Bandwidth|communication bandwidth]]** is higher because we must send a multicast with all cache line to all the PE.

In general, for these reasons, invalidation is better, in some case instead we use technique that mix the two versions of solutions.

### Event Notification with CC
Consider the following code snippet of the [[Event Notification]] implementation with shared boolean flags. Assume CC based on [[Invalidation-based Mechanism]]
```
shared bool event = false;
P … notify(event):: { event = true; };
Q … wait(event):: { RETRY: if event then event = false else goto RETRY; };
```
Once the event word is in $C_Q$ (read from $C_P$ or $M$) if **false** the event is **repeatedly tested in $C_Q$ without additional transfers** until it is **invalidated** by **P** (when P executes the `notify`)

![[Pasted image 20250521163606.png | 400]]

### Spin-Lock with CC
So far, we have assumed that the semantics of the INDIV bit is demanded to ehe **Memory Macro-Modules (MM)**. However, in **coherent [[Shared Memory Architectures|shared-memory systems]]**, atomic sequences are at the granularity of cache lines and implemented by caches with the CC firmware protocol of the machine (ie, through **CC [[Locking]] mechanisms**)

![[Pasted image 20250521164901.png]]

- **Initial condition 1**: the lock is stored in one cached **L** available in M only (value **green**)
- **Lock acquisition by $PE_i$**: $PE_i$ successfully acquired the lock that is **red** in its cache (**M** might not be updated)
- **Lock attempt by $PE_j$**: TSL is like a STORE, it invalidates L copy in $C_i$. $C_j$ acquires the **exclusive ownership** of the block L, it does not respond to CC request for L during the execution of the TSL
- **Unlock by $PE_i$**:  the STORE requires to load the cache line from $PE_j$, which does not answer if it is running the sequence in the TSL. Once the cache line has been transferred to $PE_i$ and invalidated in $PE_j$, $PE_i$ changes the value of the lock to **green**. During the next attempt to acquire the lock by $PE_j$, it will receive the ownership of the line again and it finds it **green** now.

### [[False Sharing Problem]]

# References