**Data time:** 13:35 - 14-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Collective communications]] [[Data Parallelism]]

**Area**: [[Master's degree]]
# All-to-All

Another interesting and quite complex collective is the **all-to-all collective (A2A)**, which implements a very general communication patter. It can be used to emulate different collectives (ess **[[Scatter]]** with one source only, **[[Gather]]** with one destination only)

![[Pasted image 20250514140941.png]]

- The figure above shows an example with n=3 and m=3
- Each modules on the left partitions its data structure into m partitions to be **scattered** to the modules on the right
- Each module on the right builds the final data structure assembled (**gathered**) using the partitions received from the modules on the left.
# References