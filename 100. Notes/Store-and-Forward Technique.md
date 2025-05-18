**Data time:** 23:34 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Store-and-Forward Technique

Each packet must be **completely received** by an intermediate node (switch) before forwarding the packet in output. The **transmission of a packet is not pipelined**. It requires **buffering capabilities** is each switch to hold an entire packet.

![[Pasted image 20250518233716.png | 450]]

A switch transmits all the flits of a packet to the next switch based on the [[Routing|routing algorithms]]. Each switch incorporate a buffer (**small memory**) to hold a whole packet.

### Latency
Evaluation of the latency to transmit a packet with the store-and-forward technique. Assuming the [[RDY-ACK Transmission]] mechanism. Let $\tau_{net}$ be the clock cycle length of switch units in the network, $S$ the number of flits per packet, and $T_{tr}$ the link transmission latency.

![[Pasted image 20250518234521.png | 500]]

Latency to transfer a whole packet from a switch to the next one is:
$$(S-1) \cdot (2\tau_{net} + 2T_{tr}) + (\tau_{net} + T_{tr}) = (2\cdot S - 1)(\tau_{net} + T_{tr})$$
Packet transmission is not pipelined. So, the switch $i$ starts delivering a packet to the next hop when the last flit from switch $i-1$ has been received. Latency proportional to the distance $d$ (ie, the number of units in the path):
$$(d-1) (2\cdot S -1)(\tau_{net} + T_{tr}) \sim O(d \cdot S)$$

# References