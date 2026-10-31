---
Data: 2026-10-30T23:58:00
Tags:
  - note
  - youngling
Connection:
  - "[[Searching String by Prefix]]"
  - "[[Algorithm Engineering]]"
Area:
  - "[[Master's Degree.base]]"
---
# Interpolation Search

To speak about this technique for prefix search in a dictionary of string we need first some assumption:
- Let's consider the case of a dictionary of strings that have bounded length, say shorted than $b$ char drawn from an alphabet of size $\sigma$
- We have to interpretate these strings as integers in a universe of size $\sigma^b$, for example, Una stringa di $b$ caratteri $S = c_1 c_2 c_3 \dots c_b$ viene convertita nell'intero $y$ calcolando: 
$$
y = c_1 \cdot \sigma^{b-1} + c_2 \cdot \sigma^{b-2} + c_3 \cdot \sigma^{b-3} + \dots + c_b \cdot \sigma^0
$$
We want to search the prefix $P$ into the dictionary $D$, or, searching for the **lexiographic position** of string $P$ in $D$, with the encoding above this means searching for the integer coding of $P$ among the ordered set of integers encoding the dictionary of strings.
1. First of all let $X[1, m] = x_1, \dots, x_m$ be  the array of sorted integers encoding the dictionary strings, so that $m = n$
2. We evenly subdivide the range $[x_1, x_m]$ into $m$ bind $B_1, \dots, B_m$ of size 
$$
b=\frac{x_m - x_1 +1}{m}
$$
3. We havce as many bins as dictionaru strings, and:
$$
B_i = [x_1 + (i-1)b, x_1 + ib]
$$
	in order to guarantee the constant-time access to these bins, we keep an additional array $S[1, m]$ of pointers to the first and last integers of $B_i$ in $X$.

![[Pasted image 20261031003357.png|582]]

Now given a string $P$ to be lexicographically search among the dictionary strings we:
1. Compute it's corresponding integer, let's say $y$
2. Determine the index $j$ of the candidate bin where $y$ could occur:
$$
j = \lfloor \frac{y - x_1}{b} \rfloor + 1
$$
3. [[Binary Search]] the position of $y$ in $B_j$ by accessing the subarray of $X$ delimited by the two pointers stored in $S[j]$

These two steps take $O(1 + \log|B_j|) = O(\log b)$ in **time complexity**, but to analyze in detail it's behavior we need to know some properties of this approach.

First of all let's define the parameter $\Delta$ as the **ratio between the maximum and the minimum gaps between two consecutive integers of the input dictionary**:
$$
\Delta = \frac{\max_{i=2 \dots m}(x_i - x_{i-1})}{\min_{i=2 \dots m}(x_i - x_{i-1})} = \frac{\text{gap massimo}}{g}
$$
ove $g = \min_{i=2 \dots m}(x_i - x_{i-1})$ è il **gap minimo**. This ration is essential to state the following theorem:

> [!success] Theorem
> Interpolation Search executed over a dictionary of $m$ integers takes $O(\log \min\{\Delta, m\})$ time and $O(m)$ extra space in the worst case.

> [!success] Proof
> **First Step (Relazione tra gap massimo e ampiezza del bin $b$)**
> First we want prove that the maximum number of integers that can belong to any bin is no more than $b/g$, where $g$ is defined as the minimum gap between consecutive integers of $X$.
> 
> La media aritmetica dei gap tra elementi consecutivi è: 
> $$
> \text{Media dei gap} = \frac{\sum_{i=2}^m (x_i - x_{i-1})}{m-1} = \frac{x_m - x_1}{m-1}
> $$
> Poiché la media di una serie di numeri non può mai superare il valore massimo della serie stessa, vale la disuguaglianza: 
> $$
> \max_{i=2 \dots m}(x_i - x_{i-1}) \ge \frac{x_m - x_1}{m-1}
> $$
> Applicando la proprietà aritmetica $\frac{a'}{a''} \ge \frac{a'+1}{a''+1}$ (valida per $a' \ge a'' > 0$, dado che $x_m - x_1 \ge m-1$), otteniamo:
> $$
> \max_{i=2 \dots m}(x_i - x_{i-1}) \ge \frac{x_m - x_1 + 1}{m} = b
> $$
> Quindi, **il gap massimo tra elementi adiacenti è sempre maggiore o uguale all'ampiezza del secchio** $b$
> 
> **Second Step (Limite massimo sul numero di elementi in un secchio)**
> Poiché due elementi consecutivi nell'array distano **almeno** $g$ (gap minimo), in un secchio di ampiezza $b$ possono rientrare al massimo: 
> $$
> |B_j| \le \frac{b}{g}
> $$
> Sostituendo la disuguaglianza $b \le \max_{i}(x_i - x_{i-1})$ trovata al Passo 1, si ottiene: 
> $$
> |B_j| \le \frac{b}{g} \le \frac{\max_{i=2 \dots m}(x_i - x_{i-1})}{\min_{i=2 \dots m}(x_i - x_{i-1})} = \Delta
> $$
> Poiché nessun secchio può contenere più di tutti gli $m$ elementi totali dell'array, vale il limite rigido: 
> $$
> |B_j| \le \min{\Delta, m}
> $$

# References