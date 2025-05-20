**Data time:** 01:01 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# Notification with IPI

An alternative implementation to [[Notification with Shared Flags]] of event notification is based on **I/O messages** through **[[Local IO]]** of the PE (it requires **a fixed pinning of processes/threads onto PEs**)

**Example**:
![[Pasted image 20250520011023.png|150]]

It is implemented as follows (an event may have associated data, a few data words)
![[Pasted image 20250520011102.png | 400]]

- The **notify** is implemented by **MMIO STORE**, two for the **event** and the **dest identifier**, and one for each **data word** (if any)
- In the **wait** primitive, we take care of two situations
	- **Synchronous notification**: every time PE_P executes a **notify** for an event, the destination PE_Q has already executed the corresponding **wait** on the same event.
	- **Asynchronous notification**: no **temporal relation** between **wait** and **notify** on the same event. Eg **notify might precede the wait**.


According to the specific use of the notification mechanism, we may be sure that the destination PE is waiting for an event when the source PE generates the notification. Pseudo-code with two writes (**event code** and **destination identifier**)

![[Pasted image 20250520011706.png | 300]]

Notification is triggered by writing in the memory of the UC in the source PE. This happens using standard STORE (**MMIO**). The **wait** primitive is implemented by putting the destination PE in a special state (it waits for an IPI). We assume this can be done with a `waitint` D-RISC instruction.

### Asynchronous Notification
In the **asynchronous case**, the IPI can be generated while the destination PE is not waiting for it yet (or it might run another process/thread not related to the event itself). The event must be **registered** in memory.

We assume one **flag** in memory for each event. The flags are not shared and not protected by [[Locking]]. It is possibile to use a **counter** per event, to have more notifications before a wait.

![[Pasted image 20250520012242.png | 600]]
# References