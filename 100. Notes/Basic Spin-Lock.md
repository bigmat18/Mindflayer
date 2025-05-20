**Data time:** 23:43 - 19-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Basic Spin-Lock

We introduce below a simple assembler **D-RISC** implementation of a spin-lock semaphore. For this we provide a **C-like pseuso-code** and **D-RISC compiled version**

![[Pasted image 20250519234606.png]]![[Pasted image 20250519234525.png]]

First **D-RISC compilation** with traditional LOAD/STORE (**Rlock** contains the address of the lock flag in memory). This implementation is **not correct** because the sequence containing the LOAD and the STORE is **not indivisible**.

Two PEs might load the value of the lock at the same time, and they can find the actual value equal to green. So they might acquire the lock by writing red both.

### Spin-Lock with Annotations
First correct spin-lock implementation in our D-RISC using LOAD/STORE instructions with **set_indiv** and **reset_indiv**.

![[Pasted image 20250519235056.png | 350]]

We read the current value and we atomically set it to **red**. Then, we test the old value, if it was **green** we got the lock otherwise we retry. (immediately or later on).

The LOAD (with **set_indiv**) and the STORE (with **reset_indiv**) are executed atomically. The [[Level-based view for parallel computing|firmware level]] guarantees that no other memory accesses can be executed between those LOAD and STORE by any PE.

### Spin-Lock with Delay
If the lock semaphore is **red** (value 0), the processor repeats the atomic sequence of LOAD and STORE. This can generate a high number of atomic sequences that may increase **memory contention**. To alleviate this problem, it is better to add a **delay** between two atomic sequences.

![[Pasted image 20250519235743.png | 320]] ![[Pasted image 20250519235755.png]]

**Rtimeout** contains the value of the **delay** expressed in conventional time units. Special instructions can alternatively be used to add some **nops** in the processor, like the **pause** instruction in **x86** machines.
# References