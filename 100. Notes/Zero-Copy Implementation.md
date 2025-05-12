**Data time:** 23:13 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Zero-Copy Implementation

Inter-process communication consists of two important phases:
- **Message copy**: from the message variables to the target variable
- **Synchronization**: using **context switching** or **busy waiting**

In the basic implementation of [[Channels in Message Passing|channels]] we have two copies (which is still an optimistic case, the copies might be much more, as happens with [[TCP|TCP/IP]] in distributed systems)

![[Pasted image 20250511232141.png | 350]]

Below, the pseudo-code of a zero-copy implementation of a **synchronous channel**:

![[Pasted image 20250511232932.png]]

- *wait*: boolean indicating that one process is waiting
- *length*: message length
- *val_ref*: shared pointer to the message or to the target variable
- *PCB_ref*: shared pointer to the waiting process PCB

The channel descriptor is now without the internal buffer of messages.
### Indirect Shared References
With this implemeation we have a problem, a **shared** data structures $S_0$ contains **references** to other **shared** data structures $S_1, \dots, S_n$. For example we have:
- **Ready list**: is a shared data structures that contains the **references** to the PCBs of the ready process (they are in turn shared)
- **Channels descriptor**: contins **references** to the message\target variable and the PCB of the sender/receiver (they are also shared by the processes)

![[Pasted image 20250511233443.png | 300]]

In the image above the black square box in the figure contains a sort of reference to msg, which must be meaningful for both S and R.

#### Static Method
###### Coinciding Logical Addresses
$S_i$ (msg in the example above) is referred using the same logical address by all processes (rigid exploitation of virtual memory). This solution is important because it is adopted by the **Linux Kernel**.

![[Pasted image 20250512115407.png | 500]]
The translation of the logical addresses in the **grey location** above is the same for all the processes in the system. These logical addresses are used to access the **runtime support data structures** only (PCB, ready-list, channel, descriptors, messages, target variables)
###### Distinct Logical Addresses
The reference to a generic $S_i$ (msg in the example below) is a unique identifier. Each process P has a **private table** used to translate the unique identifier to the corresponding logical address of $S_i$ in its address space.

![[Pasted image 20250512115842.png]]

As in the previus example, the use of the virtual memory is still rigid. The VM of a process must be prepared by the compiler/linker/launcher in a such a way as to statically incorporate all possible shared variables if te [[Level-based view for parallel computing|RTS]]
#### A Dynamic Method
An elegant solution to implement references to shared objects. A **capability** is an abstract data type with the following operations:
- **Acquisition**: Process P acquires the capability, i.e. it can refer the shared object by acquiring it in its logical addressing space.
- **Release**: Process P releases the capability, i.e. it cannot refer anymore the shared object with is no longer present in its logical addressing space.

![[Pasted image 20250512121312.png | 550]]
# References