**Data time:** 01:05 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory and Local IO]]

**Area**: [[Master's degree]]
# Inter-Processor Interrupts

Inter-processor interrupts are often asynchronous events used in several scenarios.
- **Processor synchronization** (e.g., asymmentric synchronization, or some notify-based spin-lock implementations for mutual exclusion might use inter-processor interrupts)
- **Low-level scheduling** (inter-processor interrupts can be used to execute scheduling functionalities on cores where no processes/threads are actually running, or to implement preemption)

**Example** of a asymmetric synchronization among two PEs.
1. The destination PE enters a spin-loop (a busy-waiting while loop until a shared flag becomes true). The source PE write true in the flag to signal the notification of an event.
2. The same might be implemented using local I/O by waiting explicitly for an IPI by the destination PE (through a so-called ```waitint``` instruction or similar), and by issuing an IPI in the source PE.

[[Local IO|UC]]’s registers (at least two, but often more) are **memory mapped**. The corresponding physical addresses are the same for each UC (so each PE can access only registers in its UC)
### Generating an IPI
Let **PEi** be the source generating and IPI directed to PEj. The message is wirtten by PEi in the UC (usually a few registers). This is done with **special instructions** or, more often, with standard LOAD/STORE (**Memory Mapped I/O**). Once a data word (ess 32 bits) is written in one specific location mapped onto the UC, the IPI is generated (ess. the **interrupt command register** in Inter APIC)

![[Pasted image 20250518013620.png | 400]]

### Receiving an IPI
The [[Local IO|UC]] stores the **message header** and the **payload** in its **local memory**, and then an **interrupt** is generated to **PEj**. **PEj** checks the presence of interrupts at the end of the interpretation of each instruction. The **firmware handling phage** of interrupt will save the message from the local memory of the UC in some registers of PEj. The next steps depend on the interrupt type (**interrupt vector**)

![[Pasted image 20250518014024.png | 400]]
# References