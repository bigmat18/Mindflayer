**Data time:** 18:39 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Cache Coherence Problem

Assume that $PE_i$ and $PE_j$ transfer the same cache line S, ie, a **shared line**, from the main memory M into their respective caches $C_i$ and $C_j$ (suppose one cache per PE)

![[Pasted image 20250520184123.png | 250]]

If **S** is **read only**, no coherency problem arises. If $PE_i$ modifies (at least one world of) **S** in $C_i$ then the S copy in $C_j$ becomes not coherent. There are **multiple** (three) **physical copies** of the same memory location (ie, of that cache line)

###### Example 1
This is an example of computation with [[Event Notification|event notification]]. Let S be initialized to 3 in M, and F(V, 3) = 5
```
P:: { S = F(V, S); notify(ready); }
Q:: { wait(ready); R = G(S); }
```

Sequence of meaningful events of (M, $C_P$, $C_Q$)
- **LOAD S by P**: miss, the cache line is read from M into $C_P$ (S=3)
- **STORE S by P, sync (write_back)**: the line is modified in $C_p$  (S=5), the modified line is copied into M (write-back)
- **LOAD S by Q**: miss, the cache line is read from M into $C_Q$ (S = 5), this is **correctly updated**.

![[Pasted image 20250520185041.png | 150]]

**No cache coherence problem arises in this example**

###### Example 2
Consider this new code fragment below
```
shared int S = 3;
P:: { wait(go); S = F(V, S); notify(ready); }
Q:: { R = G1(W, S); notify(go); wait(ready); R = G2(R, S); }
```
Sequence of meaningful events:
- **LOAD S by Q**: miss and cache line transfer from M into $C_Q$ (S=3)
- **LOAD S by P**: miss and cache line transfer from M into $C_P$ (S=3)
- **STORE S by P, sync (write_back)** line modified in $C_P$ (S=5) and copied into M (write-back)

After the notification, the memory contains the updated value. However, $C_Q$ still contains the **old value**. $C_Q$ is not correctly updated.

### Main-Copy Semantics
**Problem**: The behavior of the **logical system** is different from the one of the **physical system**.

![[Pasted image 20250520190600.png | 450]]

The semantics of a parallel computation working on shared variables **must be the same** of the computation **with the single copy of shared variables in main memory only** (ie without caches). We call this the **main copy semantics**.

**Cache coherence** tries to hide the existence of multiple copies and let the system behaves as the logical view. In the logical system:
- for each memory location L, there is exactly one copy of the value that is in M only.

Consider all LOADs and STOREs on L happened:
- At most one STORE can updated the value of L in any moment. So, there is a **total order** of STORE to L
- Let us call this total order $WR_1, WR_2m \dots, WR_n$
- The notion of **last write to a location** is globally well defined
- A LOAD of L returns the value written by some $WR_i$ in the total order. This means that LOAD is ordered after $WR_i$ and before $WR_{i+1}$

![[Pasted image 20250520191347.png | 450]]

Cache Coherence means to provide the same semantics in a system with multiple copies of L.

![[Pasted image 20250520191525.png | 250]]

**Definition**: a memory system is coherent iff is behaves as if for any given memory location L:
- There is a total order of all STOREs to L. So, all writes to the same location are serialized
- If $RD_j$ (a LOAD of L) happens after $WR_i$, it returns the value of $WR_i$ on the one of any write ordered after $WR_i$ and before $RD_j$
- If $WR_i$ happens after $RD_j$, it does not affect the value returned by $RD_j$

What does happens after above mean?

### CC in Uni-Processors
Uni-processor systems (ie, featuring **one sigle-cored CPU**) are still affected by cache coherence issues, and in case, why? The answer is yes, and because interaction with I/) units may generate coherency issues like in multi-processors.

![[Pasted image 20250520193014.png]]

- **Case 1**: Processor writes in a buffer in main memory, Processor tells the network interface card (NIC) to asynchrously send the buffer content. **Problem** is NIC might transfer stale data if processor’s writes (reflected in cached copy of data) are not flushed to memory

- **Case 2**: NIC receives a message and copies it in a buffer in main memory using DMA transfers NIC notifies CPU that the message was received, and the buffer is ready to read. **Problem** is CPU may read stale data if addresses updated by network card happen to be in cache.

**Some solutions:** processor writes to shared buffers using **uncached STOREs**, or such memory pages are marked as used by the I/O and flushed in memory explicitly by the processor before interacting with the I/O.

However, standard LAOD/STORE are much more frequent than I/O interactions. Why not have **one single shared cache** directly accessible by all PEs? 

![[Pasted image 20250520193315.png | 300]]

One single cache shared by all processors: 
- Eliminates the problem of replicating data in multiple caches
- Facilitates **fine-grained sharing** (overlapping working sets) 
- LOADs/STOREs by one processor might **pre-fetch lines** for another processor

Obvious  there are **scalability problems** (since the point of a cache is to be local and fast)

### Automatic vs Manual Solutions
The existing solutions are base on two opposite models
- **Automatic Solutions (Hardware CC)**: the problem is completely solved at the firmware level
- **Non-Automatic Solutions (Software CC)**: It is the responsibility of **programmer**, or the **developer of parallel programming libraries**, to guarantee the main copy semantics (**flushing** and **self-invaliudation** mechanisms offered at [[Level-based view for parallel computing|Assembler Level]]). Such an idea has done to handle the coherency of specific components such as the TLB. 

The first solution is widely adopted in all existing systems. It implies a **new proper semantics**, thus a specific firmware interpreter of **LOAD and STORE** instructions. Because the hardware CC sub-system is fully **[[Level-based view for parallel computing|interpretation-based]]**, in its basic definition it is not necessarily optimized for all parallel programs (i.e., more messages than the required ones might be exchanged). However, automatic solutions **make life easier to programmers**.

![[Pasted image 20250520194041.png]]
# References