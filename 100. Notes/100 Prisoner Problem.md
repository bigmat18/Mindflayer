---
Data: 2026-09-27T21:46:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# 100 Prisoner Problem

This problem is a **mathematical problem in probability theory and combinations**. The problem has different renditions in the literature. This is one of the most common:

> [!NOTE] Definition
> The director of a prison offers **100 death row prisoners**, who are numbered **from 1 to 100**, a last chance. A room contains a cupboard with **100 drawers**. The director randomly puts one **prisoner's number in each closed drawer**. The **prisoners enter the room, one after another**. Each prisoner **may open and look into 50 drawers** in any order. The drawers are **closed again afterwards**. If, during this search, **every prisoner finds their number in one of the drawers, all prisoners are pardoned**. If even **one prisoner does not find their number, all prisoners die**. Before the first prisoner enters the room, the prisoners may discuss strategy — but may not communicate once the first prisoner enters to look in the drawers. What is the prisoners' best strategy?

So in nutshell, every prisoner selects 50 drawers **independently** and **randomly**. The probability that a single prisoner finds their number is 50%. The probability that all prisoners find their numbers is the **product** of the single probabilities, which is $\bigg( \frac{1}{2} \bigg)^{100} \approx 0.8 \cdot 10^{-31}$, a vanishingly small number.

### Strategy
There is a strategy that provides for the survival of all the prisoners with probability more that 30%. 
1. The key is that the prisoners do not have to decide beforehand which drawers to open. Each prisoner can use the **information gained from the contests of every drawer they already opened to decide which one to open next**.
2. Another important observation is that this way the success of one prisoner **is not independent of the success of the other prisoners**, because they all depend on the way the numbers are distributed.

To describe this strategy we are going to numbered also the drawers form 1 to 100 (for example row by row starting from top left drawer). The strategy is the following:
1. Each prisoner first opens the drawer labeled with their own number.
2. If this drawer contains the number, they are done, and were successful.
3. Otherwise the drawer contains the number of another prisoner, and they next open the drawer labeled with this number
4. The prisoner repeats steps 2 and 3 until they find their own number, or fail because the number is not found in the first fifty opened drawers

Let's suppose that the prisoner could continue indefinitely this process, the drawers would inevitably loop back to the drawer they started with, forming a permutations cycles (like images below). By starting with their own number, the prisoner **guarantees they are on the specific cycle of drawers containing their number**. 

The question here is the following: **whether any cycle is longer than fifty drawers, and only one cycle can possibly be too long, since at most one can comprise more than half of the total drawers.**

![[Pasted image 20260928172550.png|247]] ![[Pasted image 20260928172617.png|244]]

### Example
Let's go to demonstrate that this is a good strategy to solve this problem using an example with 8 prisoners and drawers, whereby each prisoner may open 4 drawers. The distribution of the prisoners numbers is the following:

![[Pasted image 20260928174841.png]]

(in the figure **above** there is the graph representation)

The prisoners now act as follows:
- Prisoner 1 first opens drawer 1 and finds number 7. Then they open drawer 7 and find number 5. Then they open drawer 5, where they find their own number and are successful.
- Prisoner 2 opens drawers 2, 4, and 8 in this order. In the last drawer they find their own number, 2.
- Prisoner 3 opens drawers 3 and 6, where they find their own number.
- Prisoner 4 opens drawers 4, 8, and 2, where they find their own number. This is the same cycle that was encountered by prisoner 2 and will be encountered by prisoner 8. Each of these prisoners will find their own number in the third opened drawer.
- Prisoners 5 to 7 will also each find their numbers in a similar fashion

In this case all prisoners find their numbers, but it's not always the case, for example if **we swap drawers 5 and 8 would case prisoner 1 to fail after opening 1, 7, 5 and 2**.
### Permutation Representation
This probably can be described as a **permutation** of the integers 1 to 100. A sequence of numbers which after repeated application of the permutation returns to the first number is called **cycles** of the permutation. Every permutation can be decomposed into **disjoint cycles**.

> [!NOTE] Definition
> A **disjoint cycles** is a cycles which have no common elements inside.

Let's look the example above, the permutation can be written in **cycle notations** as: $(1, 8, 5)(2, 4, 8)(3, 6)$
and thus consists of two cycles of length 3 and one cycle of length 3. The permutation of the following example:

![[Pasted image 20260928175511.png]]

Can instead be written as: $(1, 3, 7, 4, 5, 8, 2)(6)$
and consists of a cycle of length $7$ and a cycle of length 1. The cycle notations is not unique since a query of length $l$ can be written in $l$ different ways depending on the starting number of the cycles.

During the opening of the drawers using the above strategy, each prisoner follows a single cycle which always ends with their own number. In the case of eight prisoners, **this cycle-following strategy is successful if and only if the length of the longest cycle of permutations is at most 4**. If a permutation contains a cycle of length 5 or more, all prisoners whose numbers lie in such a cycle do not reach their own number whithin four steps.

### Probability of Success
Now let's analyse the classic problem under the point of view of **probability**. Their survival probability is therefore equal to the probability that **a random permutation of the number 1 to 100 contains no cycle of length greater that 50**. This probability is determined in the following way:

1. A permutations of the numbers 1 to 100 can contain at most one cycle of length $l > 50$. This because if we have for example a cycle of length 51 the can't be a second $l>50$. 
2. There are exactly $\binom{100}{l}$ ways to select the number of such a cycle **(which number in this long cycle?)**
3. Within a cycle, these numbers can be arranged in $(l-1)!$ ways since there are $l$ permutations to represent distinct cycles of length $l$ because of cyclic symmetry **(in which order these number in this long cycle?)**
4. The remaining numbers can be arranged in $(100-l)!$ ways
5. Therefore, the number of permutations of the numbers 1 to 100 with a cycle of length $l>50$ is equal to
$$
\binom{100}{l} \cdot (l-1)! \cdot (100-l)! = \frac{100!}{l}
$$
The probability, that a ([uniformly distributed](https://en.wikipedia.org/wiki/Uniform_distribution_\(discrete\) "Uniform distribution (discrete)")) random permutation contains no cycle of length greater than 50 is calculated with the formula for [single events](https://en.wikipedia.org/wiki/Event_\(probability_theory\) "Event (probability theory)") and the formula for [complementary events](https://en.wikipedia.org/wiki/Complementary_event "Complementary event") thus given by

$$
1 - \frac{1}{100!} \bigg( \frac{100!}{51} + \dots + \frac{100!}{100} \bigg) = 1 - \bigg( \frac{1}{51} + \dots + \frac{1}{100} \bigg) = 1 - (H_{100} - H_{50}) \approx 0.31183
$$

where $H_n$ is the $n$-th harmonic number. Therefore, using the cycle-following strategy the prisoners survive in a surprising 31% of cases.

> [!NOTE] Definition
> In **mathematics** the n-th **harmonic number** is the sum of the reciprocals of the first $n$ natural numbers:
> $$H_n = 1 + \frac{1}{2} + \frac{1}{3} + \dots + \frac{1}{n} = \sum^{n}_{k=1} \frac{1}{k}$$

The $\bigg( \frac{100!}{51} + \dots + \frac{100!}{100} \bigg)$ part is obtained dividendo $\frac{100!}{l} / 100!$, that is the probability to obtain a cycle of length $l$

# **References**
- https://en.wikipedia.org/wiki/100_prisoners_problem