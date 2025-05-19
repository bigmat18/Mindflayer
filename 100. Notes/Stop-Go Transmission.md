**Data time:** 12:36 - 19-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Stop-Go Transmission

We assumed a simple **[[RDY-ACK Transmission]]** mechanism for sending and receiving flit in [[Wormhole Technique]], bus we can have other mechanism like **Stop/Go transmission mechanism**. 

The receiver has a buffer of $K>1$ positions (flits) and uses two thresholds (**Xon, Xoff**). Until the buffered flits are less than **Xoff**, the sender can transmit. Once **Xoff** has been reached, a **black signal** is propagated upstream. The sender restarts transmitting when the number of buffered flits becomes less than **Xon**.

![[Pasted image 20250519124604.png | 450]]

# References