**Data time:** 21:57 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Stream Parallelism]]

**Area**: [[Master's degree]]
# Basics of Message Passing

Message-passing languages or libraries provide a set of **communication primitives** to allow processes to cooperate. Different implementations of the primitives for [[Shared Memory Architectures|shared-memory]] and [[Distributed Memory Architectures|distributed-memory]]. We suppose to use typed channels. 

For example a symmetric channel CH with a sender P1 and receiver P2:

![[Pasted image 20250511220104.png]]

Send implementation copies the message value from the **message variable** into the **target variable** in the addressing space of the receiver. This is called [[Zero-Copy Implementation]] where we direct copy from the message variable to the target variable without additional copies.
### Synchronous vs. Asynchronous communications
The **asynchrony degree** of a channel (or channel capacity) is the maximum number of messages($k \geq 0$) the sender can send before it has to block waiting for the receiver to start receiving data. It depends on the memory capacity of the channel and the size of the message being sent.

- **Synchronous communications**: A send/receive operation is called synchronous if the operation completes only after the message has been received/sent. sender-receiver **rendezvous**. The sender/receiver blocks until the communication peer completes the operation
- **Asynchronous communications**: The communication operation returns immediately without  waiting for the message to be effectively sent/received
	- The completion/success of the communication will be tested later on (e.g., callbacks, futures/promises)
	- The number of asynchronous send might be limited by the asynchrony degree of the channel. Even asynchronous communication can lead to blocking if the channel buffer is finite and full
### [[Channels in Message Passing|Channels]]

### [[MP IO Non-Deterministic]]

### [[Zero-Copy Implementation]]

### [[Computation-to-Communication Overlap]]
# References