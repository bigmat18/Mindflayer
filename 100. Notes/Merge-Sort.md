---
Data: 2026-10-22T19:29:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Sorting Atomic Items]]"
Area: "[[Master's Degree.base]]"
---
# Merge-Sort

This chapter focuses on the very well known problem of sorting a set of **atomic items**

> [!Definition]
> **Atomic** means that the items occupy a constant fixed number of memory cells. Typically they are integers or reals represented with a fixed number of bytes, say four (32 bits) or eight (64 bits) bytes each.

From this let's formulate the **sorting problem**:

> [!Definition]
> The **sorting problem** is: Given a sequence of n atomic items $S[1, n]$ and a total ordering $\leq$ between each pair, sort S into increasing order.

We will consider two version of the problem. The first one is the **merge-based** paradigm, that is **merge-sort**. The focus will bet to analyze:
- the two-level memory model
- the I/O complexity

 Then we will propose some useful algortihmic tools that allow us to speed up their execution and we will demostrate that these disk-based adaptations are better .

### Sorting VS Permutation

Clearly, sorting includes permuting as a subtask. in order the sequence $S$ we need to determine its sorted permutation and then implement it (these two phases may be intricately intermingled)

> [!Definition]
> **The permutation problem** is the following: Given a sequence of n atomic items $S[1, n]$ and a per- mutation $\pi[1, n]$ of the integers ${1, 2, . . . , n},$ permute $S$ according to $\pi$ and thus obtain the new sequence $S[π[1]], S[π[2]], . . . , S[π[n]]$.

So sorting should be more difficult than permuting. So let's analyze the behavior of this two problem (sorting and permutation in the RAM model)

**Comparison-Based Sorting in RAM model**: $O(n\log n)$.
1. n array di $n$ elementi distinti può presentarsi in $n!$ (fattoriale di $n$) ordini iniziali differenti. L'algoritmo non sa a priori quale sia l'ordine corretto: il suo obiettivo è identificare **l'unica permutazione ordinata** tra le $n!$ possibili
2. Possiamo rappresentare qualsiasi algoritmo di confronto come un albero binario
	- Ogni **nodo interno** rappresenta un confronto eseguito ($A[i] < A[j]$).
	- Ogni **foglia** rappresenta una specifica permutazione dell'input.
	- Affinché l'algoritmo sia corretto e in grado di ordinare qualsiasi input, l'albero deve possedere **almeno** $n!$ **foglie** ($L \ge n!$)
3. **Worst Case**: Il numero di confronti eseguiti nel caso peggiore corrisponde all'**altezza** $h$ dell'albero (il cammino più lungo dalla radice a una foglia). Poiché un albero binario di altezza $h$ può avere al massimo $2^h$ foglie, vale la relazione: $2^h \ge n! \implies h \ge \log_2(n!)$
4. Usando l'approssimazione di Stirling per il fattoriale ($\log_2(n!) = \Theta(n \log_2 n)$)12:  $h \ge \log_2(n!) \approx n \log_2 n - n \log_2 e = \Omega(n \log_2 n)$ Poiché algoritmi come **MergeSort** o **HeapSort** riescono a ordinare un array eseguendo $O(n \log_2 n)$ confronti nel caso peggiore, il limite inferiore (_lower bound_) e quello superiore (_upper bound_) coincidono, stabilendo una complessità esatta pari a $\Theta(n \log_2 n)$

**Permuting tasks in RAM model**: $O(n)$
1. A differenza dello sorting, nel permuting **non c'è alcuna informazione da scoprire** tramite confronti: sappiamo già esattamente dove deve andare ciascun elemento. Non c'è alcun albero di decisione da esplorare.
2. Il modello RAM permette di accedere a qualsiasi indirizzo di memoria in tempo costante $O(1)$
3. L'algoritmo deve semplicemente scorrere l'array $S$ e posizionare ogni elemento nella sua destinazione $S[\pi[i]]$ (oppure effettuare gli scambi opportuni in-place)

```
permutation(arr[n], pi[n])
	for i = 0 to n:
		S'[pi[i]] = S[i];
```

Philosophically speaking, this result formally proves the intuition that moving items in the disk is the real bottleneck, rather than finding the sorted permutation. And indeed, researchers and software engineers typically speak about the I/O bottleneck to characterize this issue in their (slow) algorithms

### Merge-Based Sorting Paradigm
First of all, let's remember that owner system has:
- an **internal memory** of size M
- it allows blocked access to disk by R/W operations of $B$ items (**disk page**)

![[Pasted image 20261022202732.png|605]]

The **Mergesort** work as follow:
1. Check if  the array to be sorted consists of at least two items
	- In case it splits the input array S into two halves and then recurses on each part.
	- otherwise it is already ordered and nothing has to be done.
2. As recursion sends the two haves $S[i, m-1]$ and $S[m, j]$ are ordered and fuses (**note** that this step need an auxiliary array size $n$)

The analysis of the complexity is well know:
- the number of recursive calls is $O(\log n)$
- the merge procedure can be implemented in $O(j-i + 1)$ time using two pointers

So in the end, mergesort time complexity can be modelled as:
$$
T(n) = 2T(n/2) + O(n) = O(n \log n)
$$
### Two-Level Memory Model Analysis
The explanation above is clear and correct but let's assume that $n >M$ so it is **out-of-bound**, in this case all the computations is limited by the I/O, because we can not still use the RAM model, but we have to consider the **two-level memory model**. 
- First of all we have that **the cost of MERGE** is $O(z/B)$ I/Os with two ordered sequencies of $z$ items **in total**.
- If $M \leq 3B$ this holds: because we need 2 blocks to read in a contigui way the data, and an extra 1 block to write data
- Every time that the pointer in this procedure it a new block an I/O fault occurs, that disk page is fetched into internal memory.

So, the I/O cost to merge two sequence where the sum is $z$ is $O(z/B)$. So the total **I/O complexity** is:
$$T(n) = 2 T(n/2) + O(n/B) = O(\frac{n}{B} \log_2 n)$$

But this formula does not explain completely the good behavior of MERGESORT in practice, because it does not yet account for the **[[Cache Memory]]**

In fact, as MERGESORT recursively splits the sequence S, smaller and smaller subsequences are generated that have to be sorted. So when a subsequence of length z fits in internal memory, namely $z = O(M)$, then it is **entirely cached by the underlying operating system** using $O(z/B$) I/Os, which only accounts for the cost of loading the subsequence in internal memory.

![[Pasted image 20261022232019.png|533]]

To calculate the final complexity let's imagine that:
- L'albero di ricorsione che genera effettivi I/O su disco si arresta quando la dimensione delle foglie raggiunge $M$.
- Il numero di livelli di ricorsione sul disco è quindi $\log_2(n/M)$ (anziché $\log_2 n$).
- In ciascuno di questi $\log_2(n/M)$ livelli, la somma di tutti i sotto-array copre l'intero dataset $n$, richiedendo un totale di $O(n/B)$ I/O per livello

Moltiplicando il costo per livello per il numero di livelli otteniamo la complessità del **Binary MergeSort**: $$\Theta\left(\frac{n}{B} \log_2 \frac{n}{M}\right) \text{ I/O}$$
This bound is particularly interesting because it relates the I/O complexity of MERGESORT not only to the **disk-page size B** but also to the **internal-memory size M**, and thus to the caching feature made available to the sorter by the underlying operating system.
# References
- [[Pearls_of_Algorithm_Engineering_Ferragina_2023.pdf#page=56]]