**Data time:** 21:20 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Pipelined Communications

Firmware messages (requests/replies messages) od different types are generated for examples
- during the interpretation of **LOAD** and **STORE** instructions
- during the transmission of **inter-processor communications** via local IO

An **example** is in the case of **cache line reading** we assume that $\sigma = 64 bytes$

![[Pasted image 20250518230150.png | 600]]

Each **firmware message** can be decomposed into multiple **packets**. Each packet is also composed of multiple **flits** (a logical entity). Each flit is physically decomposed into **phits** whose width is equal to the width of the physical links in the network.

![[Pasted image 20250518230336.png | 500]]

**Flow control** is the strategy to manage network resources (ess links/buffer) between senders and receivers. For the sake of simplicity **we assume 1 flit equal to 1 phit**.

### [[Circuit Switching]]

### [[Packet Switching]]

### [[RDY-ACK Transmission]]

### [[Store-and-Forward Technique]]

### [[Stop-Go Transmission]]

### [[Credit-based Mechanism]]

### [[Wormhole Technique]]

# References