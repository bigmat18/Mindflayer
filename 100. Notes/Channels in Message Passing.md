**Data time:** 22:06 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Basics of Message Passing]]

**Area**: [[Master's degree]]
# Channels in Message Passing
###### Asynchronous Degree
The Asynchronous degree of a channels is a integer $K\geq 0$ stating the maximum amount of messages that can be transmitted by the sender before the receiver is willing to receive the first message.
### Synchronous Channels
**Syncrhronus channels** have $K=0$, this means that the channels is **memoryless**. A randezvous between send and receive temporal instants is required to complete the message transmission with $K=0$

![[Pasted image 20250511223910.png]]
- When **send** is complete, we are 100% sure that the corresponding **receive** primitive has been completely executed
- When a **receive** primitive is complete, we are 100% sure that the message received is the one produced by the last send primitive on the same channels.

### Asynchronous Channels
If K>0, the channel is **asynchronous** and allows more messages to be transmitted while the receiver is not willing to receive the first one. This decouples the message transmission from the reception. 

![[Pasted image 20250511224350.png | 500]]

If the maximum number K of unreceived messages has been reached after a send, the sender process **waits** the next receive primitive to be executed by the receiver process.

### Communication Forms
Message-passing many time has **[[MP IO Non-Deterministic|non-deterministic]]** situations because provide a large set of communication primitive and channels, each involving different number of sending/receiving processes.

![[Pasted image 20250511225858.png | 500]]

- Different implementations of the channels for each specific case
- **Multi-sender channels**: asynchrony degree K for each sending process

### Channels Implementation
We discuss the implementation of communication channels on [[Shared Memory Architectures|shared-memory systems]]. The first implementation is valid for both **synchronous** and **asynchronus channels**. For this reason, no important optimization can be applied.

The **channel descriptor** can be implemented as follows:

![[Pasted image 20250511230849.png | 500]]

The pseudo-code of the first send and receive implementation (for **multi-programmed systems**) is the following:

![[Pasted image 20250511231014.png | 550]]

The code of send and receive must be executed in a **indivisible manner** (protecting it with locks). At any time instant at most on process can be inside the send/receive code
# References