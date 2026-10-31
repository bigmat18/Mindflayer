---
Data: 2026-10-30T14:07:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Set Intersection]]"
Area:
  - "[[Master's Degree.base]]"
---
# Mutual Partitioning

This is another algorithm to solve [[Set Intersection]] problem with the assumption that the two sequence are sorted. This approach is based on **partitioning** which is the one we used to deseign [[Quick-Sort]]. 

Formally we have $|A| = n$ and $|B| = m$ and we assume that $m<n$ and that both are even numbers:
1. We select the median element $b_{m/2}$ of the shortest sequence $B$ as a **pivot**
2. We search for this value in the longer sequence $A$:
	- if $b_{m/2} \in A$, let's say $b_{m/2} = a_j$ for some $j$, and thus $b_{m/2}$ is returned as one of the elements of the inersection $A \cap B$
	- If $b_{m/2} \notin A$, let's say $a_j < b_{m/2} < a_{j+1}$ (where we assume that $a_0 = -\infty$ and $a_{n+1} = +\infty$)

In both cases the intersection algorithm proceeds **recursivelty** in two parts in which each sequence A and B has been split.

**Example**: Let's a look of the example below:
- The median of $B$ is 12 and it is used as the pivot for the mutual partition
- The pivot splits:
	- $A$ into two unbalanced parts ($A[1,3]$ and $A[5,12]$)
	- $B$ into two almost-halves parts ($A[1,4]$ and $B[6,9]$)
- This parts are recursively and correspondingly intersected
- Since the pivot occurs both in $A$ and B it is returned as an element of the intersection

![[Pasted image 20261030214757.png|561]]

Moreover, we note that the first part of A is shorter than the first part of B, and thus in the recursive call their role will be exchanged.

Above the pseudo code of the algorithm:
![[Pasted image 20261030215133.png|589]]

### Time Complexity

##### Worst Case
To evaluate the **Time Complexity** we need to identify the worst case, it happens when: 
1. the pivot falls outside A (i.e. $j=0$ or $j=n$). 
2. This means that one of the two parts in $A$ is empty 
3. and thus the corresponding half of $B$ can be discarded from the subsequent recursive calls.
4. So the [[Binary Search]] over $A$, costing $O(\log n)$ time, has discarded half of B

If this occurs at all recursive calls, the total number of them will be $O(\log m)$ thus inducing an overall cost of the algorithm equal to $O(\log m \log n)$ time. This happen with **unbalanced partitioning** of $A$. So in this case
$$T(n,m) = O(\log m \cdot \log n)$$
##### Balanced Case
Let's assume that we have a normal situation where the pivot $b_{m/2}$ falls inside the sequence A and consider the case where it coincides with the median element of $A$, say $a_{n/2}$. So in this specific situation the two partitions are balanced in both sequences, so the time complexity can be expressed via recursion:
$$
T(n,m) = O(\log n) + 2T(n/2, m/2)
$$
with the **base case** $T(n,m) = O(1)$ whenever $n,m \leq 1$. If we resolve this equation we obtain:
$$T(n,m) = O\left(m \left(1 + \log \frac{n}{m}\right)\right)$$
for any $m \leq n$. This is very important formulation because unify the two previous approach:
- **Quando le liste hanno dimensioni simili (**$m = \Theta(n)$**):** $$T(n,n) = O\left(n \left(1 + \log \frac{n}{n}\right)\right) = O(n(1 + 0)) = O(n)$$ Otteniamo esattamente la complessità dell'approccio **Merge-Based**.

- **Quando una lista è molto più piccola dell'altra (**$m \ll n$**):** Poiché $\log(n/m) = \log n - \log m \approx \log n$, la formula si riduce a: $$T(n,m) \approx O(m \log n)$$ Otteniamo esattamente la complessità dell'approccio **Binary Search**.


# References