**Data time:** 00:08 - 20-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Synchronization Mechanisms]]

**Area**: [[Master's degree]]
# RMW Instructions

They are special instructions incorporated in most of the instruction sets. They perform two memory accesses on the same location. Such accesses are made **atomic** by the [[Level-based view for parallel computing|firmware level]]. 

The **indivisibility bit mechanism** still exists but cannot be directly used and it is not visible at the Assembler Level. The use of RMW instructions is a more realistic choice in modern machine languages rather than permitting arbitrary LOADs/STOREs with indivisibility annotations. 

For example for **read-modify-write (RMW) instructions** we have:
- **Test&Set** $(r, a, op) \to \{r = M[a]; M[a] = val(r) ;\}$
- **Fetch&Op** $(r, a, op) \to \{r = M[a]; M[a] = op(r) ;\}$
- **Swap** $(r, a) \to \{tmp = M[a]; M[a] = r; r = tmp\}$
- **Compare&Swap** $(r, B, C) \to \{tmp = M[a];\: if\:(tmp==B) \:then\: M[a] = C; return \:(tmp == B);\}$

The above list is not complete, since other RMW instructions might be provided in some machines
# References