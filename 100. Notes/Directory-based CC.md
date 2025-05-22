**Data time:** 01:10 - 22-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Directory-based CC

The **[[Cache Coherence Abstract Architecture|GSK]]** is partitioned **by rows** and each partition (**Local State Knowledge** of **LSK**) is implemented in hardware by a component called **Directory**.

Directories are distributed among the PEs, each one maintain the GSK entries for a subset of the cache lines.

![[Pasted image 20250522012022.png | 400]]

- Natural strategy for [[NUMA - Non Uniform Memory Access]] each PE controls a set of GSK entries of its **local  memory**
- Applied to **[[SMP Symmetric Multi-Processor]]** as well -> **uniform partitioning** of GSK entries among PEs
- **Directory entry** is a full GSK entry containing for all the PEs whether the cache line is present, and its state (eg modified, updated)
- The PE owning the GSK entry for a line is called **Home Node**

### LOAD Interpreter
The interpreter of **LOADs** and **STOREs** is similar to the behavior see in [[Cache Coherence Abstract Architecture]], where each PE interacts with the **home node** of the referred cache line through point-to-point communications only. In case of a **LOAD b** it is useful to distinguish different actors
- **Requestor node (RE)**: PE (C2) executing the LOAD b instruction
- **Home node (HO)**: PE (C2) having information about b instruction PE is responsible for updating the state of line b in the **directory entry** of the line (which is a full GSK entry), and interacting with the **main memory** if required
- **Owner node (OW)**: PE (C2) having a **valid** copy of b in its own caches. Zero, one or more owners might exist.

There are different meaningful combinations:
- requestor $\neq$ home $\neq$ owner
- requestor $\neq$ home = owner
- requestor = home $\neq$ owner
- requestor = home and no owner

![[Pasted image 20250522012959.png]]
##### Case 1 -> requestor $\neq$ home $\neq$ owner
So in this case we have three roles and three distinct PEs involved.

![[Pasted image 20250522014305.png | 500]]
- **Optimization**: (i.e., lower latency): if RE can determine one OW based on requests issued in the past, the read request is done directly to OW, and in parallel to HO (HO updates LSK\[b\]).
- **Observation**: if the knowledge of the OW by RE is not updated, this is not a concern. OW replies that the request is wrong, so RE executes the normal protocol via HO.

##### Case 2 requestor $\neq$ home = owner

![[Pasted image 20250522014325.png | 500]]
- **Latency**: the same of the first case optimizations

##### Case 3 requestor = home $\neq$ owner

![[Pasted image 20250522014404.png | 500]]
- **Latency**: the same of the first case optimizations

##### Case 4 requestor = home and no owner

![[Pasted image 20250522014439.png | 500]]

The **PE** knows that it is the home node for line b. So, it knows whether the line is present in other caches. If not, it transfers the cache line directly from the **main memory**
- **Important**: the **LSK** of all the involved PEs may be enriched with possible useful information that can be used to reduce the number of interactions among caches

### STORE Interpreter
Suppose a **STORE** that performs a **C1-C2-M write-back**. Assume the line is already in the cache of that PE (**write-hit**)
##### Case 1 RE = OW $\neq$ HO
Cache line modification is executed in C1, C2 of RE. The line becomes the only **valid** copy; so RE becomes the only owner OW at the end of the protocol. **Store Notification**:
1. The change is communicated to **HO** (write-back), which knows the PEs having a valid copy (possible including HO itself),
2. and communicates **invalidation** to all of them
3. when HO receives ACKs from all them
4. forwards an ACK to RE

In case of **synchronous write-back STOREs**, the interpreter explicitly waits for that ACK. M and LSK\[b\] updating can be done in parallel with the invalidation protocol.

![[Pasted image 20250522015845.png]]

**Store Notification**: in general, it conveys the cache line to be written. The home node can update the memory immediately or later on. **The cache line content is not allocated in the HO caches**
##### Case 2 RE=HO=OW
The write-back STORE can be executed by the home node itself. Assume the cache line already in the caches of that PE.

Cache line modification is executed in C1, C2 of **HO**. The line becomes the only **valid** copy; **HO** becomes the only **OW** at the end of the distributed protocol. 
1. **HO** determines the PEs having a copy of the line, and sends **invalidations** to all them. No store notification message is generated
2. **HO** waits for the ACKs from all of them. This must be explicitly waited in case of **synchronous write-back STOREs**

M and LSK \[b\] updating can be done **in parallel** with the invalidation protocol.

![[Pasted image 20250522020548.png]]

Since the **STORE** is triggered by the home node of the cache line, no **STORE notification** is paid and the STORE is conceptually complete when all the ACK messages are received from the PEs having the copies to be invalidated.

### Synchronous STORE and CC
Director-based CC and impact on [[Event Notification]]. In the example below, if we use a normal STORE to write S, even if this STORE is annotated as `write_back`, we might start the next STORE to update `ready` when the interpretation of the previous STORE is not compete (on **weak ordering** machines)

The effect is that PE4 might receive the invalidation for `ready` and load it again finding it equal to true before the invalidation of S arrives
```
P1:: { S = 10;
notify(ready); }
P4:: { … F(S); …
wait(ready); y = G(S); }
```

![[Pasted image 20250522022434.png | 400]]

A processor P completes the execution of a **synchronous STORE**, and can start the next instruction only when the effect of that STORE is **globally visible** in the machine. The use of synchronous STORE on S (or a `FENCE` after), guarantees that when the interpretation of the STORE on `ready` starts, the privius STORE on S is visible (have been invalidated including the copy in PE4)

```
P1:: { S = 10;
notify(ready); }
P4:: { … F(S); …
wait(ready); y = G(S); }
```

![[Pasted image 20250522022745.png | 400]]

### Hierarchical CC Solutions 
In existing multiprocessor architectures (especially for [[Multicore Technologies|multi-CMP systems]]), the CC can be managed **hierarchically**:
- **Inner protocol**: it is the CC protocol inside the CMP. Usually, it is **snoopy-based** owing to the relatively low parallelism within each CMP (e.g., 4-8 PEs).
- **Outer protocol**: it is the CC protocol among CMPs, it is usually **directory-based** (not always), where the home node is a CMP (the local controllers for the outer protocol are the **shared L3 caches** of the CMPs, if they exist, or chip wrapper units, i.e., **WWs**)

![[Pasted image 20250522023038.png | 550]]
# References