---
Data: 2026-10-27T23:53:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Sorting Atomic Items]]"
Area:
  - "[[Master's Degree.base]]"
---
# Multi-Way Quick-Sort

This approach of **Multi-Way Quick-Sort** is the opposite that we can see in [[Multi-Way Marge-Sort]]. While the multi-way merge sort achive the I/O efficiency merging $k$ run each times, the Multi-way quick sort achive the same result in the opposite way: **we split an input sequence in $k=\Theta(M/B)$ sub-sequence (called bucket) using $k-1$ pivot**


> [!NOTE] Definition
> We can define the buckets set in the following way: given $s_1, s_2, \dots, s_{k-1}$ the $k-1$ pivots choose to partition the sequence $S[1\dots n]$. Let's define two "fake pivots" $s_0 = -\infty$ and $s_k = +\infty$. The i-th bucket is define as: $$B_i = \{S[j] : s_{i-1} < S[j] \le s_i\} \quad \text{per } i=1, \dots, k$$

Each partition phase can be implemented in $O(n/B)$ I/Os by using a memory organization that is the opposite of the one employed for [[Multi-Way Marge-Sort]], so, in the RAM memory we will have:
- **1 input block**: use to read sequentially the sequence $S$ to be partitioned
- **k output blocks**: allocated one for each $k$ bucket that we are creating.

Poiché $k = \Theta(M/B)$, la somma di questi $1 + k$ blocchi occupa $O(M)$ celle di RAM, rispettando perfettamente la capienza della memoria interna. Ogni volta che un blocco d'uscita si riempie con $B$ elementi, viene scaricato su disco con $1$ I/O

L'algoritmo legge il dataset dal disco blocco per blocco (tramite il blocco d'ingresso in RAM)1:
1. Per ciascun elemento letto, si determina a quale dei $k$ bucket appartiene confrontandolo con i pivot $s_1, \dots, s_{k-1}$ (tramite ricerca binaria in RAM)1.
2. L'elemento viene inserito nel corrispondente **blocco d'uscita** in RAM associato a quel bucket.
3. Quando un blocco d'uscita in RAM si riempie (raggiunge $B$ elementi), viene **scaricato (flushed) su disco** con 1 I/O sequenziale, e il blocco in RAM torna vuoto per accogliere i successivi elementi
4. Una volta completata la scansione sequenziale dell'intero dataset:
	- Sul disco si sono formati $k$ file distinti corrispondenti ai $k$ bucket.
	- L'algoritmo viene richiamato **ricorsivamente** su ciascun bucket $B_i$4.
	- **Caso Base**: Quando la dimensione di un bucket scende al di sotto della capacità della RAM ($\le M$), la ricorsione si arresta: il bucket viene caricato interamente in memoria interna e ordinato in RAM (es. con HeapSort o QuickSort in-memory) richiedendo solo $O(|B_i|/B)$ I/O per il caricamento

La complessità d'I/O totale del Multi-way QuickSort è: 
$$
O\left(\frac{n}{B} \log_{M/B} \frac{n}{M}\right) \text{ I/O}
$$
this bacause:
1. **Costo d'I/O per livello =** $O(n/B)$ **I/O**: Ad ogni livello dell'albero di ricorsione, l'algoritmo legge l'intero dataset d'input e lo riscrive distribuito nei vari bucket. Poiché sia la lettura sia la scrittura dei blocchi completi avvengono in modo sequenziale, il costo per elaborare $n$ elementi a un dato livello è di $O(n/B)$ trasferimenti di blocco
2. **Altezza dell'albero di partizionamento =** $O(\log_{M/B} \frac{n}{M})$. Grazie alla tecnica dell'Oversampling (Lemma 5.1), ciascuno dei $k = \Theta(M/B)$ bucket riceve al massimo $O(n/k)$ elementi con alta probabilità, quindi ad ogni livello ricorsivo, la dimensione dei sotto-problemi si riduce di un fattore $k \approx M/B$

	Partendo da un dataset di dimensione $n$ e fermando la ricorsione quando i sotto-problemi hanno dimensione $\le M$, il numero di livelli necessari è dato dalla risoluzione dell'equazione:
$$
\frac{n}{k^h} \le M \implies k^h \ge \frac{n}{M} \implies h = \log_k \left(\frac{n}{M}\right) = O\left(\log_{M/B} \frac{n}{M}\right)
$$

### Oversampling
Per garantire che l'albero di partizionamento abbia un'altezza minima pari a $O(\log_{M/B} \frac{n}{M})$, occorre che i $k$ bucket siano bilanciati, ovvero che ciascuno contenga $\Theta(n/k)$ elementi. To find efficiently $k-1$ good pivots we deploy a fast and simple randomized strategy called **oversampling**

![[Pasted image 20261028153555.png|627]]

We have $a$ parameter that controls the **amount of oversampling** and thus impacts of the **robustness** of the selection process as well as on the time efficiency (Step 2). The algorithm is the following:
1. Si estrae un campione casuale di $(a+1)k - 1$ elementi dall'input (con $a = \Theta(\log k)$)
2. Si ordinano questi campioni in RAM in tempo $O((ak) \log(ak))$
3. Si scelgono come pivot gli elementi equispaziati di $a+1$ posizioni nel campione ordinato

![[Pasted image 20261028160319.png|552]]

If we look in detail the choose of the parameter $a$ we can see that:
- Se $a$ è grande, i campioni forniscono una rappresentazione molto fedele della distribuzione dei dati e i bucket risultano ben bilanciati, ma cresce il costo per ordinare i campioni in RAM4.
- Impostando $a = \Theta(\log_2 k)$, si ottiene un bilanciamento quasi perfetto con un costo di selezione trascurabile pari a $O(k \log_2^2 k)$ tempo di CPU


> [!success] Theorem (Chenoff Bound)
> Siano $k \ge 2$ e $a + 1 = 12 \ln k$. Un campione di dimensione $(a+1)k - 1$ garantisce che tutti i $k$ bucket ricevano **meno di** $4n/k$ **elementi**, con probabilità almeno $1/2$.

![[Pasted image 20261028163941.png|434]]

> [!success] Proof
> To demonstrate this probability first of all let's consider the sorted version of the input sequence $S$, which hereafter we denotate by $S'$. We logically split $S'$ in $k/2$ segments of length $2n/k$ each. 
> 
> We are interest in the following event: if there exists a bucket $B_i$ with at least $4n/k$ items assigned to it, for some index $i$. This "large buffer" completely spans at least one segments (that we will call $t_2$)
> 
> By the definition of buckets, the pivots $s_{i-1}$ and $s_i$ delimiting $B_i$ fall outside $t_2$. But to build the algorithm above for the segment $t_j$ they must fall strettamente meno di $(a+1)$ campioni (altrimenti almeno uno di essi sarebbe stato scelto come pivot interno a $t_j$).
> 
> Pertanto, applicando l'Union Bound: 
> $$
> \mathcal{P}(\exists B_i : |B_i| \ge 4n/k) \le \frac{k}{2} \times \mathcal{P}(\text{un determinato segmento } t_j \text{ contiene } < a+1 \text{ campioni})
> $$
> 1. **Calcolo del Valore Atteso**: La probabilità che un singolo campione estratto a caso cada nel segmento $t_j$ è $\frac{2n/k}{n} = \frac{2}{k}$. Sia $X$ il numero di campioni che cadono in $t_j$. Il valore atteso $E[X]$ su $(a+1)k - 1$ campioni estratti è: 
$$ E[X] = ((a+1)k - 1) \times \frac{2}{k} = 2(a+1) - \frac{2}{k} \ge 2(a+1) - 1 \ge \frac{3}{2}(a+1) \quad (\text{poiché } a \ge 1)$$
>Da $E[X] \ge \frac{3}{2}(a+1)$ ricaviamo che $a+1 \le \frac{2}{3}E[X] = \left(1 - \frac{1}{3}\right)E[X]$.
>
>
> 2. **Applicazione del Chernoff Bound**: Ricordando la forma standard del Chernoff Bound per le deviazioni inferiori, $$\mathcal{P}(X < (1-\delta)E[X]) \le e^{-\frac{\delta^2}{2} E[X]}$$ 
> 	Impostando $\delta = 1/3$: 
> $$
> 	\mathcal{P}(X < a+1) \le \mathcal{P}\left(X < \left(1 - \frac{1}{3}\right)E[X]\right) \le e^{-\frac{(1/3)^2}{2} E[X]} = e^{-\frac{E[X]}{18}} 
> $$
> 	Sostituendo $E[X] \ge \frac{3}{2}(a+1)$ e l'ipotesi del lemma $a+1 = 12 \ln k$: 
> $$
> 	e^{-\frac{E[X]}{18}} \le e^{-\frac{\frac{3}{2}(a+1)}{18}} = e^{-\frac{a+1}{12}} = e^{-\ln k} = \frac{1}{k}
> $$
> 3. **Conclusione**: Sostituendo $\mathcal{P}(X < a+1) \le \frac{1}{k}$ nella disuguaglianza iniziale dell'Union Bound: 
> $$
> \mathcal{P}(\exists B_i : |B_i| \ge 4n/k) \le \frac{k}{2} \times \frac{1}{k} = \frac{1}{2} \quad \blacksquare
> $$
> 

# References