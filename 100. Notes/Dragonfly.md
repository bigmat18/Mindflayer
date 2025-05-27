**Data time:** 17:32 - 27-05-2025

**Status**: #note #youngling 

**Tags:** [[Parallel and distributed systems. Paradigms and models]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Dragonfly

This is a **Multi-Level network**. Three levels: Router, Group and System. Each router has connections to $p$ endpoints, $a-1$ local channels (to other routers in the same group) and $h$ global channels (to routers in other group)
$$
\deg(DF(p,a,h)) = p + (a-1) + h
$$
A group consists of a routers. Each group has **ap** connections to endpoints (i.e., fully  connected topology) and **ah** connections to  global channels.
$$
dia(DF(p,a,h)) = 3 \:\: or \:\: 5
$$
while the **bw** is large.

![[Pasted image 20250527173623.png | 300]]
# References