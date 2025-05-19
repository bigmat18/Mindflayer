**Data time:** 23:23 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# RDY-ACK Transmission

This is a **transmission mechanism** where each link is associated with **two control units** (RDY, ACK) from sender to receiver and back.
- **RDY=1** means that a new flit is delivered to the destination unit (ie the main link contains a new meaningful flit)
- **ACK=1** means that the receiver unit allows the sender unit to overwrite the main link content with a new flit.

The triple composed of <main_link, RDY, ACK> is called **firmware communication interface**.

![[Pasted image 20250518232709.png | 350]]

- **RDY** line should be **set** by the sender and **reset** by the receiver
- **ACK** line should be **set** by the receiver and **reset** by the sender
- RDY and ACK events are identified by any transition from 0 to 1 or from 1 to 0 of the corresponding control bit.
- Simple sequential and combinational circuits.

### Head-of-Line Blocking
In case of a network conflict, the **ack** signal is delayed and header flit of the conflicting message is kept in the corresponding input interface.

![[Pasted image 20250519122755.png | 400]]

Variant (**Virtual Cur Through**): all flits of the conflicting message are received and buffered by the **congested switch**.
# References