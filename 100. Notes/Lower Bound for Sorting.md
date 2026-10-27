---
Data: 2026-10-25T16:30:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Sorting Atomic Items]]"
Area:
  - "[[Master's Degree.base]]"
---
# Lower Bound for Sorting

When we presented [[Merge-Sort]] we spoke about the relation between the **sorting and permutation problem**, concluding that the former is more difficult than the latter in the RAM model.

Before digging into the proof of this lower bound, let us briefly show how a sorter can be used to permutate a sequence of items $S[1, n]$ in accordance with a given permutation $\pi[1, n]$.

This will allow us to derive an upper bound to the number of I/Os that suffice to solve the permuting problem for any instance $(S, \pi)$ Recall that, this means generating a sequence:
$$
S[\pi[1]], S[\pi[2]], \dots, S[\pi[n]]
$$
In the **RAM model** we can jump among $S$ imputes according to the permutation $\pi$ and create the new sequence $S[\pi[i]]$ for $i=1,2, \dots n$, thus taking $O(n)$ optimal Time. On disk we have two different algorithms:
1. The first consists of mimicking what done in RAM, paying one I/O per moved item and thus taking $O(n)$ I/O worst case
2. The second consists of genereting a proper set of tuple and the sorting them:
	1. The algorithm create the sequence $P$ of pairs $(i, \pi[i])$ where $i$ indicate the position where the item $S[\pi[i]]$
	2. Then it sorts these apirs according to the $\pi$-component
	3. and via a parallel scan of $S$ and $P$ substitutes $\pi[i]$ with the item $S[\pi[i]]$
	4. thus creating the new pairs $(i, S[\pi[i]])$
	5. Another sort is executed according to the first component of these pairs, 
	6. so that another parallel scan of $S$ and P can be used to write $S[\pi[i]]$ into $S[i]$ thus optainint the seuquence of items correctly permuted.

![[Pasted image 20261025175407.png|585]]

This **second algorithm** bring to a **time complexity** for permutation of $O(min\{n, \frac{n}{B} \log_{M/B}\frac{n}{M}\})$
n what follows we will show that this approach, in its simplicity, is I/O-optimal. The two upper bounds for sorting and permuting asymptotically equal each other whenever $n = O(\frac{n}{B} \log_{M/B}\frac{n}{M}\})$.

### Lower Bound for Sorting
In the I/O model the critical resource it's not the CPU comparison, but the **number of blocks transfers (I/O)**. 

##### Decision Tree in RAM model
First of all in a RAM model we can define a **decision tree** in the following way:
- Ogni nodo interno dell'albero di decisione rappresenta **1 singolo confronto** tra due elementi (grado di uscita pari a 2), that means $A[i] < A[j]$.
- Le foglie rappresentano le possibili permutazioni ordinate dell'input. Poiché un array di $n$ elementi ammette $n!$ permutazioni, l'albero deve possedere **almeno** $n!$ **foglie**.

If we analyze the complexity of the sorting using the tree an execution of the algorithm **is equal to a walk from root to a leaf** so, per un albero binario di altezza $h$, si ha:
$$2^h \ge n! \implies h \geq \log_2(n!) \implies h = \Omega(n \log_2 n)$$
##### Decision Tree in Two-Level model
For the **two-level memory model** is can be redefine in this way:
- Ogni nodo interno rappresenta 1 operazione di I/O.
- Le foglie dell'albero sono sempre $n!$ (il numero di permutazioni possibili dell'input). But what change is the **grado di uscita (fan-out) di un nodo d'I/O**: Rappresenta il numero massimo di risultati distinti (relazioni d'ordine) che un singolo I/O può generare.

The fan-out is more complicated to calculate: Quando facciamo 1 I/O e portiamo $B$ elementi in RAM, dobbiamo distinguere due contesti:
1. **Touched Pages**: Se la pagina sul disco era già stata letta in passato, l'algoritmo **conosce già l'ordine relativo** tra i $B$ elementi al suo interno. 
	- L'unica nuova informazione che l'I/O rivela è **come questi** $B$ **elementi si intercalano (si intrecciano) con gli** $M - B$ **elementi già presenti in RAM**. 
	- Scegliere come posizionare $B$ elementi ordinati in $M$ posizioni totali equivale a scegliere $B$ posizioni su $M$, cioè il coefficiente binomiale $\binom{M}{B}$
2. **Untached Pages (Pagine mai lette prima)**: Il dataset contiene $n$ elementi divisi in $\frac{n}{B}$ **blocchi/pagine d'input**
	- La prima volta che leggiamo una di queste pagine, l'algoritmo **non conosce l'ordine relativo** dei $B$ elementi al suo interno
	- Poiché $B$ elementi possono presentarsi in $B!$ permutazioni interne, a questa prima lettura dobbiamo moltiplicare le combinazioni di intercalamento $\binom{M}{B}$ per le $B!$ permutazioni interne dei nuovi arrivati.

Immaginiamo un qualsiasi algoritmo che risolve l'ordinamento eseguendo $t$ **operazioni di I/O** nel caso peggiore:
- $\frac{n}{B}$ **nodi** sono letture di pagine mai viste prima (_Untouched_), ciascuno con fan-out $\binom{M}{B} \cdot B!$
- I restanti $t - \frac{n}{B}$ **nodi** sono I/O su pagine già lette in precedenza (_Touched_), ciascuno con fan-out $\binom{M}{B}$

> [!success] Proof
> In un albero di decisione di profondità $t$ (dove $t$ è il numero di I/O eseguiti nel caso peggiore), il numero totale di foglie è limitato dal prodotto dei gradi di uscita lungo il cammino:
> 
>   
> 
> $$\text{Numero Massimo di Foglie} \le \left(\binom{M}{B} \cdot B!\right)^{\frac{n}{B}} \times \left(\binom{M}{B}\right)^{t - \frac{n}{B}}$$
> 
> Semplificando i termini con la stessa base $\binom{M}{B}$: $$\text{Numero Massimo di Foglie} \le \left(\binom{M}{B}\right)^t \cdot (B!)^{\frac{n}{B}}$$
> 
> Affinché l'algoritmo sia in grado di ordinare correttamente qualsiasi sequenza, il numero di foglie deve essere almeno pari a $n!$:
> 
>   
> 
> $$\left(\binom{M}{B}\right)^t \times (B!)^{\frac{n}{B}} \ge n!$$
> 
> Passando ai logaritmi e applicando l'**approssimazione di Stirling** ($\log(x!) \approx x \log x - x \log e$ e $\log \binom{M}{B} \approx B \log \frac{M}{B}$):
> 
>   
> 
> 1. $t \cdot B \log_2 \left(\frac{M}{B}\right) + \frac{n}{B} \cdot (B \log_2 B) \ge n \log_2 n$
>     
>       
>     
> 2. $t \cdot B \log_2 \left(\frac{M}{B}\right) + n \log_2 B \ge n \log_2 n$
>     
>       
>     
> 3. $t \cdot B \log_2 \left(\frac{M}{B}\right) \ge n (\log_2 n - \log_2 B) = n \log_2 \left(\frac{n}{B}\right)$
>     
>       
>     
> 
> Risolvendo rispetto a $t$:
> 
> $$t \ge \frac{n \log_2(n/B)}{B \log_2(M/B)} = \Omega\left( \frac{n}{B} \log_{M/B} \frac{n}{B} \right) \text{ I/O}$$

From this we can say that:
> [!success] Theorem
> In a two-level memory model with internal memory of size M, disk-page size B and D disks, a comparison-based sorting algorithm must execute $\Omega\left(\frac{n}{B} \log_{M/B} \frac{n}{M}\right) \text{ I/O}$

# References