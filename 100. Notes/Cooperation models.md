**Data time:** 14:00 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Introduction to HPC]]

**Area**: [[Master's degree]]
# Cooperation models

Overview of the two general models to express cooperation among parallel entities.
### Horizontal Structuring
Some levels can be represented a collection of cooperating autonomus entites generically called modules. 
- This is true for the [[Level-based view for parallel computing|OS level]] with applications implemented as set of cooperating **process/threads**
- This is true also for [[Level-based view for parallel computing|Firmware level]] with a system described as a collection of cooperating firmware units.

![[Pasted image 20250511140512.png]]

Cooperation can be expressed with two different paradigms
- **Local-environment model (message passing)**: modules have **only private resources** and cooperation can be done by exchanging values as messages through **channels** (communication primitives)
- **Global-environment model (shared variables)**: shared resources and data exist and can be manipulated by modules using **synchronization primitives**

#### Message-Passing Model
It can be used at the **OS level** and **Firmware bevel**.
###### Example

![[Pasted image 20250511141009.png]]

We have  a communication channel of type T. The first process do a **send** using the channels and the second use **receive** to use channels to take the variable. From the user perspective, each process can  access only private data.
#### Shared-Variables Model
From the user perspective, each process can access only private data. This is not suitable at the **Firmware level**.
###### Example

![[Pasted image 20250511141434.png | 500]]
# References