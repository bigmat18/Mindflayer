---
Data: 2026-10-28T18:39:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Random Sapling]]"
Area:
  - "[[Master's Degree.base]]"
---
# Random Sampling Streaming Unknown-Length

This version is different from [[Random Sampling Straming Model]] because we don't know $n$ that is a crucial point to compute the probability seen in the previous algorithm.

Let's discuss a first basic solution:
- We use a **min-heap** $H$ of size $m$
- And a random generator $Rand(0,1)$

The idea is to **associate a random priority $r_j$** to each item $S[j]$ and then use min-heap $H$ to select the items that have the top-m priorities.

![[Pasted image 20261028191026.png|573]]

In the algorithm above:
1. we extract the minimum priority from $H$
2. if it is less that the current generated priority:
3. we remove the element associate from the Heap and 
4. we add the new $S[j]$ assigning the priority $r_j$
5. In the end we return $H$ contained the $m$ high priority elements

We can conclude that:

> [!NOTE] Complexity
> Algorithm above takes $O(n/B)$ I/Os and $O(n log m)$ time, generates n random numbers, and uses $O(m)$ additional space to sample uniformly at random m items from the sequence $S[1, n$] in a streaming-like way and without the knowledge of $n$

### Reservoir Sampling
A more elegant algorithm to resolve the unknown problem is called **reservoir sampling** witch improve the algorithm above in **time** and **space complexity**.

![[Pasted image 20261028191707.png|424]]

This algorithm works in the following way:
- I use a **reservoir** array $R[1,m]$ to keep the candidate samples.
- Initially $R$ is set to contain the first $m$ items of the input sequence.
- At any subsequent step $j$ the algorithm makes a choice whether $S[j]$ has to be included or not in the current sample. This choice occur with a probability of:
$$
P(j) = m/j
$$
- When a $S[j]$ is included some previously selected item has to be kicked out of $R$, this is chosen at random in $R$ with probability $1/m$.

We will have a **double choice** by choosing an integer $h$ in the range $[i,j]$ and making the substitution only if $h \leq m$. This event has probability $m/j$, exactly what we wished to set for $P(j)$.

For the **correctness** of the algorithm:
- it's clear that the algorithm select $m$ items
- It's less clear that these items are drawn uniformly at random from $S$

The second point means that the element are drawn with probability $m/n$. Let's try to demonstrate this.

> [!success] Proof
> Vogliamo dimostrare che, al termine dell'analisi di $n$ elementi, ogni elemento $S[j]$ (per $1 \le j \le n$) risiede nel serbatoio $R$ con probabilità esatta pari a $m/n$.
> - **Caso Base (**$n = m$**)**: I primi $m$ elementi vengono copiati direttamente in $R$. Ciascun elemento ha probabilità $m/m = 1$ di essere nel serbatoio. L'induzione è verificata.
> - **Passo Induttivo (da** $n-1$ **a** $n$**)**: Assumiamo per ipotesi induttiva che dopo il passo $n-1$, ciascun elemento $S[1 \dots n-1]$ appartenga a $R$ con probabilità $\frac{m}{n-1}$.
> 
> Al passo $n$ arriva l'elemento $S[n]$:
> 1. **Probabilità per l'ultimo elemento** $S[n]$: $S[n]$ viene inserito in $R$ se e solo se $h = \text{RAND}(1, n) \le m$. Poiché $h$ è scelto uniformemente in $[1, n]$, questa probabilità è esattamente: 
> $$
> \mathcal{P}(S[n] \in R) = \frac{m}{n}
> $$
> 2. **Probabilità per un elemento precedente** $S[j]$ **(con** $j < n$**)**: L'elemento $S[j]$ risiede nel serbatoio dopo $n$ passi se e solo se: 
> $$
> \text{Risiedeva in R al passo n-1} \:\:\: AND \:\:\: \text{Non viene rimosso al passo } n.
> $$
> 
>La probabilita che risieda in R al passo n-1 e $\frac{m}{n-1}$ per ipotesi induttiva, mentre la rimozione **non avviene** in due casi indipendenti:
>
> 1. $S[n]$ non viene selezionato per entrare nel serbatoio (avviene con probabilità $1 - \frac{m}{n}$).
> 2. $S[n]$ viene selezionato, ma la posizione da rimpiazzare $h$ NON è la posizione occupata da $S[j]$ (avviene con probabilità $\frac{m}{n} \times \frac{m-1}{m}$).
> 
> Sommando le probabilità dei due casi di mantenimento: 
> $$
> \mathcal{P}(\text{Mantenuto}) = \left(1 - \frac{m}{n}\right) + \left(\frac{m}{n} \times \frac{m-1}{m}\right) = 1 - \frac{m}{n} + \frac{m-1}{n} = \frac{n - m + m - 1}{n} = \frac{n-1}{n}
> $$
> 
> Pertanto, la probabilità totale per $S[j]$ al passo $n$ è: 
> $$
> \mathcal{P}(S[j] \in R \text{ al passo } n) = \mathcal{P}(S[j] \in R \text{ al passo } n-1) \times \mathcal{P}(\text{Mantenuto}) = \frac{m}{n-1} \times \frac{n-1}{n} = \frac{m}{n} \quad \blacksquare
> $$
> 

# References