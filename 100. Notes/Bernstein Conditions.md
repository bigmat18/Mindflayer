**Data time:** 01:42 - 13-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Bernstein Conditions

He **domain** of a task is the set of variables that the task reads during its execution; the **codomain** in the set of variables potentially modified by the task execution.

There is 3 main conditions:
- **read-after-write (RAW)** dependency $C_1 \cap D_2 = \emptyset$, i.e., the second function reads variables whose value is produced by the first function.
- **write-after-read (WAR)** dependency $C_2 \cap D_1 = \emptyset$, i.e., the second function modifies variables that are read by the first function
- **write-after-write (WAW)** dependency $C_1 \cap C_2 = \emptyset$ i.e., there exist variables that are modified by the execution of both $F_1$ and $F_2$.

The third conditions WAW do not have effect in some synchronous computational models (e.g., at the [[Level-based view for parallel computing|firmware level]], with variables implemented by clocked registers). Only the RAW is valid at any system level for any language semantics. The second WAR is not present in functional languages because provided the concepts of variable assignment.
# References