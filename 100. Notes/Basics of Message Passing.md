**Data time:** 21:57 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Basics of Message Passing

Message-passing languages or libraries provide a set of **communication primitives** to allow processes to cooperate. Different implementations of the primitives for [[Shared Memory Architectures|shared-memory]] and [[Distributed Memory Architectures|distributed-memory]]. We suppose to use typed channels. 

For example a symmetric channel CH with a sender P1 and receiver P2:

![[Pasted image 20250511220104.png]]

Send implementation copies the message value from the **message variable** into the **target variable** in the addressing space of the receiver. This is called [[Zero-Copy Implementation]] where we direct copy from the message variable to the target variable without additional copies.
### [[Channels in Message Passing|Channels]]

### [[MP IO Non-Deterministic]]

### [[Zero-Copy Implementation]]
# References