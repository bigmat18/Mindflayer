**Data time:** 23:20 - 17-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Memory and Local IO]]

**Area**: [[Master's degree]]
# Memory Controllers (MINF)

MINFs in the CPU convert the physical address of the cache line into a **bank identifier**, **row identifier**, **column identifier**. They are on-chip units in the CMPs doing **scheduling** of memory requests from the LLC cache(s).

A basic policy is **FR-FCFS** (**first-ready**, **first-come-first-serve**): MINF prioritizes accesses to the currently open rows first, and then serves other requests to other rows in FIFO order. MINF can coalesce multiple requests into one larger request (to take advantage of **[[Dynamic RAM (DRAM)]] burst mode**).

![[Pasted image 20250517232325.png]]

**Single-channeled MINF** communicates with one rank at a time. Other schemes are possible (**dual-channeled**, **quand-channeled**). MINFs have a complex internal structure to keep track of
resources to prevent conflicts.
# References