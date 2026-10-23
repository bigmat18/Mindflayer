---
Data: 2026-10-22T23:35:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Sorting Atomic Items]]"
Area: "[[Master's Degree.base]]"
---
# Multi-Way Marge-Sort

This is an important optimization of the standard [[Merge-Sort]] that allow to optain the best I/O complexity. In the classic merge-sort implementation we merge two parts in one, this behavior impacted on the base of the logarithm (base 2), we want try to increase this number.

This is possibile and useful in terms of I/O because, when we do a merge operations we are going to use 3 block of size B (3B memory), 2 for the two parts, 1 for the final output. But there could be cases where the **internal memory can store more data** $M/B >> 3$ so we want use this memory in the current level to reduce I/O complexity

So, we will design a k-way partition of the merge sort, with $k = (M/B)-1$, so:
- that $k$ blocks are available to read block-wise $k$ input sorted runs
- and one (the -1) will be available for writing

> [!error] Problem
> The challenge is the following: at each step we have to select the minimum among k candidate items from k distinct sorted runs without iterate over each $k$ value (time complexity of $O(k)$)

The smarter solution is using a [[Tree Min-Max|Min-Heap]] data structure and takes $O(\log k)$ time per item written to the output block:
- The min-heap will contains $k$ pairs (one per input run)
- Each consisting of two components `<item, index>`
- Initially the item will be the minimum items of the $k$ runs
- It will be taken and insert in the final output block
-  we also extract the item in `index+1` in the same runs and we insert it

![[Pasted image 20261023000244.png|503]]

Fondere $k$ run di lunghezza totale $z$ richiede quindi $O(z/B)$ I/O e $O(z \log k)$ tempo di CPU.  The final **I/O complexity** can be calculated in this way:
- Costruendo un albero di fusione a $k$ vie con $O(n/M)$ foglie (le run ordinate iniziali)9:
- Il numero di passaggi completi sui dati (altezza dell'albero) è $O\left(\log_{M/B} \frac{n}{M}\right)$.
- Ogni passaggio legge e riscrive l'intero dataset pagando $O(n/B)$ I/O
- La complessità d'I/O totale del **Multi-way MergeSort** risulta essere: $$ O\left(\frac{n}{B} \log_{M/B} \frac{n}{M}\right) \text{ I/O} $$
> [!warning]
> In some cases we can express this **I/O complexity** in the following way. Usiamo le proprietà dei logaritmi per scomporre l'argomento: $$ \log_{M/B} \left(\frac{n}{B}\right) = \log_{M/B} \left(\frac{n}{M} \cdot \frac{M}{B}\right) = \log_{M/B} \left(\frac{n}{M}\right) + \log_{M/B} \left(\frac{M}{B}\right) $$
Poiché $\log_{M/B} \left(\frac{M}{B}\right) = 1$, otteniamo: $$ \log_{M/B} \left(\frac{n}{B}\right) = \log_{M/B} \left(\frac{n}{M}\right) + 1 $$
In notazione asintotica, l'aggiunta di una costante $+1$ nell'argomento non modifica la classe di crescita8: $$ \log_{M/B} \left(\frac{n}{M}\right) = \Theta\left(\log_{M/B} \frac{n}{B}\right)$$
So in the end we will have: $$
\Theta\left(\frac{n}{B}\log_{M/B} \frac{n}{B}\right)
$$

This no make any difference asymptotically but in practice  the number of merging levels will be very small.



# References
- [[Pearls_of_Algorithm_Engineering_Ferragina_2023.pdf#page=63]]