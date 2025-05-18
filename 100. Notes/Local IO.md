**Data time:** 00:52 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory and Local IO]]

**Area**: [[Master's degree]]
# Local IO

A **local I/O sub-system** is often present within a CMP. It consists of on-chip local I/O units one per PE, which are used for generating/receiving **[[Inter-Processor Interrupts]] (IPI)**. A local I/O per PE (we call it **communication unit - UC**). IPI might convey one or a few data words as a payload.

![[Pasted image 20250518010328.png | 400]]

- UC might be equipped with **DMA** (rarely) and **MMIO** (common) in more complex scenarios
 - Writing in some registers of the UC (e.g., the **interrupt vector** and the id of the destination PE) will trigger the inter-processor interrupts
- The on-chip network used to interact with MINFs can be used for IPI too (or a dedicated on-chip network might be present, **interrupt bus**)

### Multi-CMP Local I/O
Local I/O can be extended to cover **multi-CMP** architectures. IPI directed to a PE in another CMP should cross the **external off-chip network**. IPIs cross the CMP boundaries through the external on-chip interfaces (e.g., QuickPath/UltraPath on Intel, Infinity Fabric on AMD) through one of the available **EXT-INFs** (chosen in a load-balanced manner or using other policies).

![[Pasted image 20250518014140.png]]

### Asynchronous Notification
A process A on **PEi** generates an [[Inter-Processor Interrupts]] composed by the **event_code** and bye two data words (**data_1, data_2**). The message is sent to PEj.

The IPI might arrive while PEj is running any process/thread that might be not directly interested to that interrupt. The firmware handling phase will jump to the execution of an **handler** (routine) saving the IPI and its payload in a **queue data structure**.

![[Pasted image 20250518014531.png | 500]]

### Synchronous Notification
Like for asynchronus notification we have a process A on **PEi** generates an [[Inter-Processor Interrupts]] composed by the **event_code** and bye two data words (**data_1, data_2**). The message is sent to PEj.

It is possible that on the destination **PEj** there is a process B explicitly waiting for the IPI (i.e., so in a a special idle state until an interrupt of a given type is signaled by its **UC**)

![[Pasted image 20250518015023.png | 500]]

No interrupt handler execution needs to be executed in this case, since the actions to handle the IPI are implemented by the instructions following the EI in the code. The code (right) is just an example. The spin-loop waiting for the IPI can be implemented in different manners
# References