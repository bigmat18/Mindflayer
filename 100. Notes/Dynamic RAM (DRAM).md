**Data time:** 23:03 - 17-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory and Local IO]]

**Area**: [[Master's degree]]
# Dynamic RAM (DRAM)

- Made with cells that store data as a charge on **capacitors**
- Require periodic charge **refreshing** to maintain data storage
- Used to implement the **off-chip main memory** of a system (it is large and cheap, with high density)
- Typically capacity in the order of **hundreds of GiB or some TiBs**
- Latency for read and write in the order of **tens/hundreads of nanoseconds**
### DRAM Array
Ia a DRAM array bits are stored as electric charges (one **transistor** plus one **capacitor** per bits cells). It is composed of a given number of **row lines** and **column lines**. A **decoder** is a circuit putting one of the output wires to one based on the input address (the other are set to zero)

![[Pasted image 20250517225539.png | 400]]

- logically, each row contains **2K bits** grouped into **256 bytes** (1byte = 8 bits)
- Based on the row address, the given row is **charged** (the one of the red squares in the figure), and we are ready to read any byte in that row
- An **analog-to-digital converter** (not shown in the picture) converts charge on bit-lines to digital values stored in the row buffer
- The column selection logic provides **8 bits** in output corresponding to the selected byte (the red squares in the figure)

Reading a byte from a DRAM array requires different steps having a [[Communication Latency]] each: **pre-charge, row activation, columns selections phases**, plus the transfer to the requesting PE

![[Pasted image 20250517225945.png]]
Pre-charge (**PRE**) is the process of "closing" the currently active row by transferring the contents of the row buffer back into the active row's capacitors and resetting the bit-lines

![[Pasted image 20250517230148.png]]
To improve performance, DRAM accesses are always in **burst mode**. Burst bytes are transferred but discarded when accesses are not for sequential locations.

### Banked DRAM Chips
A single **DRAM chip** will contain multiple DRAM Arrays (also called **banks**), to enable **pipelined interleaved processing** of memory requests. While one bank is busy handling the activation or precharge steps, another bank can be actively transferring data.

![[Pasted image 20250517230615.png | 500]]

- Precharge/activate row/columns selection to one bank while transferring data from another bank
- All banks of a DRAM chip **share the same output pins** (one byte transferred at a time)
- The goal is in all cycles, the chip is sending one byte, **highest [[Processing Bandwidth|bandwidth]] for a DRAM chip**

### DIMM Macro-Module (MM)
To further increase bandwidth, we can place more DRAM chips in the same **DIMM macro-module**. Eight DRAM chips represent a rank. A single memory request for 8 contiguous bytes (64 bits) is served by a rank.

![[Pasted image 20250517230943.png | 550]]

**Memory bandwidth**: approximation is $B_M = (fHz \cdot 64 bits)/8$ in bytes per second ess, a **DDR** at 400 MHz has $B_M \sim 3.2 GiB/s$. Modern DDR5 archives about **50 GiB/S** (bandwidth calculation takes other parameter into account not shown here)

### Reading Cache Line
PEs generate read/write requests to the memory on a **cache line basis** (64 bytes in most of the machines, and for all the levels of the cache hierarch
##### Bad Idea
A first approach is to assign the whole line to the same DRAM chip. Even with banking, we can transfer at most one byte per cycle of the memory.

![[Pasted image 20250517231651.png]]

Bytes are sent consecutively over the same pins.
##### Good Idea
A good idea is to interleave the addresses of the bytes in the line and distribute them to the different DRAM chips of a rank. Each DRAM chip produces 8 bytes of the line (they can be interleaved in the internal banks)

![[Pasted image 20250517231745.png | 550]]

64 bytes are sent consecutively over all the pins of the rank.
# References