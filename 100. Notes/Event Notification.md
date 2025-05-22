**Data time:** 00:54 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Event Notification

We need specific primitives for asymmetric synchronization between processors. Two primitives: **wait** (for an event) and **notify** (of an event).

![[Pasted image 20250520005601.png | 500]]

- **Example** of pure precedence relation without sharing data.
	![[Pasted image 20250520005646.png|500]]

- **Example** of asymmetric synchronization with shared data.
	![[Pasted image 20250520005731.png | 500]]

In general more the one asynchronous event
![[Pasted image 20250520005834.png | 500]]

We will restrict the use of event notification: i.e., for each event, we admit at most one **notify** primitive executed before the corresponding **wait** primitive.
### Notification Pattern
How can we be sure that at most one **notification** is performed before **wait**? This depend on the way in which notifications are used in the program.

**Example**: [[RDY-ACK Transmission]] pattern between processes/threads. 
- P1 semds a notification of an event **RDY**.
- Before doint that, it waits for the notification of an event **ACK** by P2

![[Pasted image 20250520012601.png | 400]]

When can this pattern be useful? **Lock-free** implementation of a **message-passing communication channel**.Initialization: `idxS, idxR to zero, RDY[i]=false, ACK[i]=true`

![[Pasted image 20250520012650.png]]

### [[Notification with Shared Flags]] 

### [[Notification with IPI]]
# References