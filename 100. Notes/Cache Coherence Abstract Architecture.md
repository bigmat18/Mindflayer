**Data time:** 17:35 - 21-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Cache Coherence]]

**Area**: [[Master's degree]]
# Cache Coherence Abstract Architecture

CC synchronization implies a **centralization point**. In particular, a **Global State Knowlage** (GSK) about **shared cache lines** is conceptually needed.

![[Pasted image 20250521173728.png | 400]]
- Which lines are **shared**
- Which lines are currently in the caches, and of which PEs
- What is the **state** of the line in each PE (valid, invalid, modified, ...). The kinds of state depend on the specific **CC protocol** implementation

We devise an **Abstract Architecture for CC**, where a logical centralized entity (**Global Controller, GC**) is in charge of managing the **GSK**, interacting with the various PEs. This is an hypothetical architecture to understand the problems.

In the **Abstract Architecture for CC**, we have the following entities that interact during the interpretation of LOAD and STORE instructions:
- **Main memory** (conceptually one single entity, not true actually)
-  **Global Controller** (conceptually it is a single unit, not possible in a real-world implementation for scalability issues)
- **Processing elements**, each having one cache (which acts as a **Local Controller**)
- **[[Cache-to-Cache (C2C) Transfers|cache-to-cache interconnection network]]**

![[Pasted image 20250521174540.png]]

###### LOAD
Interpreter of LOAD b executed by $PE_0$ (b = cache line identifier)
- **b is in $C_0$** -> no CC action (local copies are always identifier)
- **if miss**, GC is properly informed. if one $C_j$ contains b (eg the one of PE1), GC delegates $PR_1$ to transfer the cache line sto $PE_0$ via [[Cache-to-Cache (C2C) Transfers]], Otherwise (currently, no b copy in cashes), GC transfers the cache line from M to $PE_0$
- **GSK is updated in both cases**

###### STORE
Interpreter of a STORE b executed by $PE_0$. Assume b is already in cache of $PE_0$, otherwise the STORE interpretation executes the actions of a LOAD with miss. Then, it executes "atomically" the following actions:
- **b** is modified locally in $C_0$ and **GC is properly informed**
- GC updates M (not necessarily, it can be done later on) and it si charge of communicating the **invalidation messages** to all the b copies (in $PE_i, PE_{j1}, \dots, PE_{jk}$) 
- When alla $PE_i, PE_{j1}, \dots, PE_{jk}$ reply with a **done** GC replies to $PE_0$
- **GSK is updated**

### CC firmware implementations
In real systems, GC must be distributed among the local controllers (LCs). Two solutions:
- [[Snoopy-based CC|Snoopy-based architecture]]: the GSK is conceptually **partitioned by column**
- [[Directory-based CC|Directory-based architecture]]: the GSK is conceptually **partitioned by rows**

The former needs a **centralization point** (ie, actually a broadcast medium, eg a **[[Buses|Bus]]**). The latter adopts **point-to-point communications** directed to the PEs having a copy of the considered cache line, and to the PE having the corresponding GSK entry. This approach works in any network connection PE-Caches-Memory.
- Snoopy-based solutions are the de-facto standard for **low-level parallelism in CMPS** (2-8 cores)
- Directory-based CC is applied to **high-parallelism CMPs**

Hybrid hierarchical solutions are often applied to [[Multicore Technologies|multi-CMP systems]]

# References