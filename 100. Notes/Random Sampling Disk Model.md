---
Data: 2026-10-28T17:06:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Random Sapling]]"
Area:
  - "[[Master's Degree.base]]"
---
# Random Sampling Disk Model

To start let's assume some constraints:
1. We start by assuming that the input size $n$ is known and that $S[i,n]$ is stored in a contiguous set of pages on disk.
2. These pages cannot be modified because they may be the input of a more complicated problem that includes the current one as a sub- task

![[Pasted image 20261028171257.png]]

The algorithm is very simple and it's doing the following things:
1. At each step s the algorithm maintains the following invariant: **the subarray $S'[n-s + 1, n]$ contains the items that have already been sampled, and the unsampled items of S are contained in $S'[1, n-s]$**
2. Initially (s=0) this invariant holds because $S'[n-s + 1, n] = S'[n+1, n]$ is the empty array
3. In the next steps $s$, the algorithm selects uniformly at random one item from $S'[1, n-1]$ and replaces it with the last item of the sequence. This preserve the invariant for $s+1$
4. At the end (ie $s=m$), the sampled items are contained in $S'[n-m + 1, n]$

This algorithm has a set of issue:
- $S'$ cannot be a **pure copy** of $S$, but it must be implemented as an array of pointers to $S'$s items. The reason is that these items may have variable length, so their retrieval in constant time could not be obtained via arithmetic operations
- the **replacement step might be impossible due to a difference in length** between the item at $S'[p]$ and the item at $S'[n-s]$. Pointers avoid this issue but occupy $\Theta(n\log n)$ bits of extra space.
- Another problem is the **pattern for the memory access** that is pure random over $O(n)$ cell, thus taking $O(m)$ I/Os

Now let's try to improve this algorithm in terms of **I/Os** and **space** resources.

![[Pasted image 20261028172457.png|635]]

This algorithm is more simple but it has some important aspect to be careful:
1. The choose of the **Dictionary implementation** is essential in the time complexity of the algorithm, exists many differents way to do that. We have choosen this: 
	Implementando $\mathcal{D}$ tramite una **Hash Table** di dimensione $\Theta(m)$ con liste di trabocco (chaining) e una funzione di hash universale, ogni ricerca ed inserimento richiede $O(1)$ tempo atteso e uno spazio totale di soli $O(m)$
2.  **Resampling (Gestione Duplicati)**: Se viene estratta una posizione $p$ già presente in $\mathcal{D}$, l'estrazione viene scartata e ripetuta.
	- La probabilità di estrarre un duplicato è $\frac{|\mathcal{D}|}{n} \le \frac{m}{n} < \frac{1}{2}$ (assumendo senza perdita di generalità $m < n/2$; se $m > n/2$, si risolve il problema complementare selezionando gli $n-m$ elementi da non campionare)10.
	- Di conseguenza, il numero atteso di tentativi di estrazione per ogni nuovo elemento è costante ($\le 2$)

With this consideration we can conclude that:
> [!NOTE] Fact
> The algorithm above on hashing with chaining requires $O(m)$ expected time and $O(m)$ additional space to select uniformly at random $m$ positions in $[1, n]$. 

And also:
> [!NOTE] Fact
> An additional sorting step is needed if we wish to extract the sampled items of S in a streaming-like fashion. In this case, the overall sampling process takes $O(min\{m, n/B\})$ I/Os


# References