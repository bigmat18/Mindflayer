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
#### Coinciding Logical Addresses
#### Distinct Logical Addresses
#### A Dynamic Method
# References