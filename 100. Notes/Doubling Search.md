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
# Doubling Search

Despite the optimal time complexity of [[Mutual Partitioning]] it remains still heavily based on recursive calls and binary searching, which are two paradigms that offer **poor performance in a disk-based setting**

In order to improve this algorithm we introduce the **Doubling Search** or **Galloping Search** of also **Exponential Search**.

![[Pasted image 20261030223302.png|495]]

Let's image that the first $j-1$ elements of $B$ are already checked in $A$:
- so $b_{j-1}$ is located immediately after $a_i$ in sorted $A$
- this means that $a_i \leq b_{j-1} < a_{i+1}$
- so when we check the next elements in $B$ (named $b_j$) it suffices to search for it in $A[i + 1, n]$

The **idea** of the algorithm comes here, instead of binary searching this sub-array, we execute **doubling search** that means that we check elements in $A[i+1, n]$ at distance that grow as power of 2 so:
1. we compare $b_j$ with $A[i+2^k]$ for $k=0,1,\dots$
2. until we have one of this two situation:
	- for some $k$ we find a value $b_j < A[i+2^k]$
	- of we jumped out of the array $A$, so $i+2^k > n$
3. Finally we perform a binary search for $b_j$ in $A[i + 1, \min \{i + 2^k, n\}]$
4. and we return $b_j$ if the search have success
5. We repeat recursively the same approach in the sub-array $A[1, i']$ with $i' = $ the indices returned by binary search.

![[Pasted image 20261030231806.png|590]]

For the **Time Complexity** evaluation let's denotate:
- $i_j = i'$ the position where $b_j$ in $A$
- $i_{j-1}$ the position of $b_{j-1}$ in $A$
Clearly $i_{j-1} \leq i_j$. We also set $i_0 = 0$ and denotate with $\Delta_j = \min\{2^{k-1}, n\}$ the size of the sub-array where the binary search of $b_j$ is executed.

From the condition of the while loop the position $b_j \in A$ such that:
$$i_j - i_{j-1} \ge 2^{k-1} \text{ and } i_j < \min\{i_{j-1} + s^k, n\}$$
Where we have:
- $i_j - i_{j-1} \ge 2^{k-1}$ means that $b_j$ is larger than the previously checked element in $A$
- $i_j < \min\{i_{j-1} + s^k, n\}$ means either $b_j$ is smaller than $A[i_{j-1} + 2^k]$ or that checked position is out of $A$

We can therefore write:
$$i_j - i_{j-1} \ge 2^{k-1}$$
Combinando questa disuguaglianza con la definizione di $\Delta_j$, otteniamo la relazione fondamentale: $$\Delta_j \le 2^{k-1} \le i_j - i_{j-1}$$
At this points we have all the ingredients to estimate the total length of the searched subarrays of $A$. Sommando le dimensioni di tutte le finestre $\Delta_j$ per tutti i $j = 1, 2, \dots, m$, otteniamo una **somma telescopica** in cui i termini intermedi si cancellano a vicenda:
$$\sum_{j=1}^{m} \Delta_j \le \sum_{j=1}^{m} (i_j - i_{j-1}) = (i_1 - i_0) + (i_2 - i_1) + \dots + (i_m - i_{m-1}) = i_m - i_0$$
Poiché $i_0 = 0$ e $i_m \le n$, la somma totale di tutte le ampiezze delle finestre non può mai superare la lunghezza totale di $A$: $$\sum_{j=1}^{m} \Delta_j \le n$$
Il costo totale dell'algoritmo su tutti i $m$ elementi di $B$ è: $$\text{Tempo Totale} = \sum_{j=1}^{m} O(1 + \log \Delta_j) = O\left( m + \sum_{j=1}^{m} \log \Delta_j \right)$$

Poiché la funzione logaritmo $f(x) = \log x$ è una **funzione concava**, possiamo applicare la **Disuguaglianza di Jensen**:

$$\frac{1}{m} \sum_{j=1}^{m} \log \Delta_j \le \log \left( \frac{1}{m} \sum_{j=1}^{m} \Delta_j \right)$$

Sostituendo la relazione $\sum_{j=1}^m \Delta_j \le n$ trovata al punto C:

$$\frac{1}{m} \sum_{j=1}^{m} \log \Delta_j \le \log \left( \frac{n}{m} \right) \implies \sum_{j=1}^{m} \log \Delta_j \le m \log \left( \frac{n}{m} \right)$$

Sostituendo questo risultato nella formula del tempo totale:

$$\text{Tempo Totale} = O\left(m + m \log \frac{n}{m}\right) = O\left(m \left(1 + \log \frac{n}{m}\right)\right)$$

# References