**Data time:** 23:49 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Wormhole Technique

Based on the **header**, the switch decides the output link, ess **OUT1**. The header contains the number of flits of the payload (**LEN**). So, it remembers that all payload flits from **IN1** must be routed to **OUT1**. Until the transmission is not complete, **OUT1** cannot be used by other packet transmissions. If two worms of flits should use the same output interface in a switch, we have a **network conflict**.

![[Pasted image 20250519000217.png | 500]]
### Latency
Assuming the [[RDY-ACK Transmission]] mechanism. Let $\tau_{net}$ be the clock cycle length of switch units, S the number of flits per packet, and $T_{tr}$ the link transmission latecy.

**Example** with $d=4$ and any S. The following diagram holds if we have one phit per flit.

![[Pasted image 20250519000530.png | 400]]
The latency of the packet transmission can be evaluated as:
$$(2 \cdot S + d - 3)(\tau_{net} + T_{tr})$$
Difference with [[Store-and-Forward Technique]] is remarkable. In terms of orders of magnitude, we have $O(d + S)$ instead of $O(d \cdot S)$. What happens if we have 2 phits per flit?


# References