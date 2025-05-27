**Data time:** 20:50 - 18-05-2025

**Status**: #note #youngling 

**Tags:** [[High Performance Computing]] [[Interconnecton Networks]]

**Area**: [[Master's degree]]
# Fat-Tree Networks

A tree structure can be used to implement an **[[Indirect Networks]]** where endpoints are the leaves while switch units are the intermediate nodes. For example a **binary trees** reduce the switch degree at the expense of latency compared with **non-binary trees**

![[Pasted image 20250518210138.png | 500]]

**Network conflicts** become high in the switches at higher levels, because they interconnect two separate sub-trees.

### Common Ancestor Routing
Let B be $A \oplus B$ with A source and B destination, binary identifiers. Let i be the position of the first bit equal to 1 in R starting from the most significant big (ie the leftmost one). Route up to the switch having level i, which is the **least common ancestor**. Down in the direction given by the bits of B from bit i to the least significant bit (**1** means **right**, **0** means **left**).

![[Pasted image 20250518210555.png]]
###### Example 1 (A to B)
R = 011, so we have to reach the switch at level 1. Then we consider the last two bits of B, so we go right and then right again.
###### Example 2 (A to C)
R = 110, so we have to reach level 0, the root. Then we consider the bits of C (110), so we go right, right and in the end at left.

### Fat-Tree
Introduced by Charles Leiserson (1985). The idea is to keep the bandwidth constant at each level (e.g., we have eight links at each level in the picture below).

![[Pasted image 20250518210852.png | 500]]

**Conflicts can be sharply reduced**, and approximately their effect can be **neglected** in the under-load latency evaluation. Unfortunately, the structure above is not realistic and cannot be implemented as is. Switches at level $k = 1 \dots n-1$ (with $n = \log_2 N$) or the tree are monolithic $2^{n-k} \times 2^{n-k}$ **[[Crossbars]]**

The main **problem** is that the cost and complexity of switches increases from level to level making infeasible to build a large Fat Tree connecting many nodes. 

A possibile solution is to map a Fat Tree topology with **N leaves** directly onto the switches of a **k-ary n-fly network** ($k^n = N$)

![[Pasted image 20250518211315.png | 250]]

Now each switch has a **limited degree**, so its design is feasible. Only modest increase in contention owing to the modular design of switches. Suitable for [[NUMA - Non Uniform Memory Access]] and [[SMP Symmetric Multi-Processor]] if switches behave according to [[Butterfly (2-fly n-ary) Networks|butterfy routing]] or to the **tree routing** depending on the type of messages.

For a 2-level Fat Tree where each switch has k ports:
- $k + \frac{k}{2}$ switches in total
- $n = k \times \frac{K}{2}$ maximum number of endpoints
- $bw(n) = \frac{n}{2}, \deg(n) = 4, dia(n) = 4$


# References