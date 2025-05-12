**Data time:** 22:05 - 11-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Basics of Message Passing]]

**Area**: [[Master's degree]]
# MP IO Non-Deterministic

Very often message-passing programs have non-deterministic behaviors.
##### Input Non-Deterministic
D receives and processes input messages from any client C[\*] sequentially. However it serves one input message coming from any C[\*] as soon as it is available, regardless the specific client ready to send it.

![[Pasted image 20250511225146.png]]

In the example above there are programming primitives to probe the state of a set of channels to discover non-empty channels. We assume to have a probability to 0.333 to receive each of possibile inputs.

##### Output Non-Deterministic
 S transmits each produced message to one of its destination processes D[\*] randomly chosen by selecting a destination whose channel is not full. If all channels are full S waits.

![[Pasted image 20250511225449.png]]

In the example above there are programming primitives to probe the state of a set of channels to discover non-empty channels. We assume to have a probability to 0.333 to send each of possibile outputs.
# References