---
Data: 2026-10-25T21:36:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Sorting Atomic Items]]"
Area:
  - "[[Master's Degree.base]]"
---
# Quick-Sort

Like [[Merge-Sort]] also **Quick-Sort** is based on **[[Divide & Conquer]] paradigm**, so it proceeds by dividing the arrya to be sorted into two parts, which are then sorted recursively.

But unlike MERGESORT, QUICKSORT does not explicitly allocate extra working space, its combine step is absent, and its divide step is sophisticated and impacts on its overall efficiency.

![[Pasted image 20261025213810.png|509]]

> [!NOTE] Key Idea
> The key idea is to partition the input array $S[i, j]$ into two parts such that one c**ontains items that are smaller than the items contained in the latter part**. This partition is order-preserving because no subsequent steps are necessary to recombine the ordered parts after the two recursive calls.

**Partitioning** is typically obtained by selecting one input item as a pivot, and by distributing all the other input items into two subarrays according to whether they are smaller/greater than the pivot (step 4).

> [!NOTE] Note
> In the pseudocode the pivot is forced to occur in the first position $S[i]$ of the array to be sorted, this is obtained by swapping the chosen pivot $S[r]$ with $S[i]$ before the procedure `PARTITION(S, i, j)` is invoked.

There are two issues for achieving efficiency in the execution of QUICKSORT:
1. one concerns the implementation of PARTITION(S, i, j)
2. and the other the relationship between the size of the two formed parts, because the more balanced they are, the more QUICKSORT comes closer to MERGESORT and thus to the optimal time complexity of $O(n\log n)$

### Three-Way Partitioning
The goal of `PARTITION(S, i, j)` is to divide the input array into two parts, one containing items that are smaller than the pivot, and the other containing items that are larger than the pivot. Items equal to the pivot can be arbitrarily distributed between the two parts.
- At the end of PARTITION(S, i, j), the pivot is located at $S[p]$, 
- the smaller items are stored in $S[i, p − 1]$
- and the larger items are stored in $S[p + 1, j]$

This partition can be implemented in many ways, taking O(n) optimal time, but each way offers a **different cache usage** and thus **different performance in practice**.

Now we consider a version that takes into account the presence of **items equal to the pivot**. They are detected and stored aside in a “special” subarray which is located between the two smaller/larger parts. t is clear that the central subarray, which contains items equal to the pivot, **can be discarded from the subsequent recursive calls, similarly to how we discard the pivot.**

![[Pasted image 20261025220057.png|544]]

Following a detail description of the steps of the algorithm:
1. We use 3 pointers (l, r, c)
	- $l$: delimita l'inizio della regione degli elementi uguali a $P$
	- $r$: delimita l'inizio della regione degli elementi maggiori di $P$
	- $c$: l'indice corrente dell'elemento sotto esame ($S[c]$)
2. Per ciascun elemento $S[c]$ incontrato durante la scansione, l'algoritmo distingue **3 casi**
	1. **Se** $S[c] > P$: Non occorre effettuare scambi: basta far avanzare $c$
	2. **Se** $S[c] == P$: Si scambia $S[c]$ con $S[r]$ e si incrementa $r$ ($r++$)4. In questo modo l'elemento uguale a $P$ viene inserito alla fine della regione centrale, facendo scorrere a destra la regione degli elementi maggiori
	3. **Se** $S[c] < P$: - Si eseguono **due scambi consecutivi**:
	    - `swap(S[c], S[l])`
	    - `swap(S[c], S[r])`
		Si incrementano sia $l$ che $r$ ($l++, r++$). Questo fa avanzare il confine degli elementi minori e fa scorrere a destra l'intervallo degli elementi uguali e maggiori

![[Pasted image 20261025220836.png|511]]

this reduces the number of items to be sorted recursively, but needs a change in he (classic) pseudocode because **PARTITION must now return the pair of indices which delimit the central subarray instead of just the position p of the pivot**.

The three-way partitioning algorithm takes O(n) time and offers two positive properties:
1. [[Stream Parallelism|strem]]-like access to the array S, which allows the processor to prefetch the items to be read;
2. the items equal to the pivot, residing in the subarray S[l, r − 1], can then be eliminated from the following recursive calls because they are in their final correct position.
# References