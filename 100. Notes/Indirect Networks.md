**Data time:** 17:51 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Indirect Networks

Indirect networks provide multiple **stages** to reach all destinations from a given source endpoints. Each stage is composed of a set of [[Interconnection devices|switches]]. Therefore, some switches are directly connected to endpoints (i.e., the ones in the **first** and the **last** stage), while others are connected to other switches only (i.e., all the **intermediate stages**). Therefore, multi-staged networks are **indirect** by definition.

![[Pasted image 20250518185443.png | 450]]

Basic structure of a **bidirectional switch unit** that we will use in network topologie. Below an **Implementation of a unidirectional monolithic 2x2 [[Crossbars]]**

![[Pasted image 20250518185743.png]]        ![[Pasted image 20250518185750.png]]
###### Bidirectional Monolithic Switch
A **switch** implements a **[[Crossbars]]** with parallelism at the clock cycle level. It is composed of two units, one for routing messages in one direction, and the others on the opposite one. Possibile only for limited values of **N** for **pin count** and design **complexity** reasons

The idea of **limited-degree networks** is to implement more complex switches NxN (with large N) as a composition of interconnected smaller switches (e.g., 2x2)
### [[Butterfly (2-fly n-ary) Networks]]

### [[Omega Networks]]

### [[CLOS Networks]]

### [[Fat-Tree Networks]]

### Butterflies and Fat-Tree in SMP
**Butterflies** and **Fat Trees** can be used in large **multi-CMP [[SMP Symmetric Multi-Processor]] and architectures**. They can jointly implement the **P-M network** and the **P-P network**.

![[Pasted image 20250518211746.png | 350]]

### Butterflies and Fat-Tree in NUMA
**Butterflies** and **Fat Trees** can be used in large **multi-CMP [[NUMA - Non Uniform Memory Access]] of SMPs architectures**. They can jointly implement the **P-M network** and the **P-P network**.

![[Pasted image 20250518211853.png | 350]]

# References