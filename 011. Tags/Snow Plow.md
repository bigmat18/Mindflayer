---
Data: 2026-10-23T00:11:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Sorting Atomic Items]]"
Area: "[[Master's Degree.base]]"
---
# Snow Plow

If we look the complexity of the [[Merge-Sort]] we have $O(\frac{n}{B}\log \frac{n}{B})$ I/O complexity (or in case of [[Multi-Way Marge-Sort]] we have $O(\frac{n}{B}\log_{M/B} \frac{n}{M})$).  We can notice that the larger $M$  is the smaller is the number of merge-passes over the data. So to improve the algorithm we can move on this direction. We can do that in the following way:
- Buy more RAM (not an option for us)
- Work on compression algorithm on integer
- Apply algorithm techniques as **Snow-Plow**

So, let's go for the third. The name **Snow-Plow** of derive from the idea that:

> [!NOTE] Idea
> Uno spazzaneve che avanza pulendo una strada circolare mentre continua a nevicare:
> - I fiocchi di neve che cadono **davanti** allo spazzaneve vengono spazzati via subito durante il tragitto corrente (_elementi che possono entrare nella run attuale_). 
> - I fiocchi che cadono **dietro** allo spazzaneve dovranno attendere il giro successivo (_elementi rimandati alla run successiva_).
>
>In practice the tecnique: cans the input sequence S and generates sorted runs whose length have variable size which is never smaller than M, and proved to be 2M on average

To apply this approach we use a **different sorting scheme** with some consideration:
1. it first **creates these sorted runs**, now of variable length, and **then applies repeatedly over these sorted runs the MERGE** procedure. Although runs will have different lengths, the **MERGE** will operate as usual, requiring an optimal number of I/Os for thier merging.
2. A second thing is that we asume that **items are transferred one at a time from disk to memory**, rather than block-wise. Eventually, since tha algorithm scans the input items, it will be arrarent that the numer of I/Os required by this process is still the optimal $O(n/B)$ I/Os

Below there is the pseudo-code of the algorithm:
![[Pasted image 20261025143406.png|395]]

These are the steps:
1. **Inizializzazione della fase**:
	- Si caricando $M$ elementi non ordinati dalla sequenza di input e si costruisce il min-heap $H$ di dimensione $M$. (this can be done in a **linear time complexity** of $O(M)$)
	- L'array ausiliario $U$ viene inizializzato come vuoto ($|U| = 0$)
2. **Ciclo principale di scansione**: Finche il min-heap $H$ non è vuoto:
	1. **Estrazione e Scrittura**: Si estrae il valore minimo da $H$ (chiamato `min`) e lo si scrive sul disco nel file della _run_ corrente
	2. **Lettura**: Si legge il prossimo elemento `next` dall'input non ordinato (that means we read a new element from $S$, so the external input, not $M$)
		- **Se** $next \ge min$: L'elemento `next` è maggiore o uguale all'ultimo valore appena scritto su disco. Può quindi far parte della _run_ corrente senza romperne l'ordine crescente, perciò viene inserito in $H$
		- **Se** $next < min$: Scrivere `next` nella _run_ corrente violerebbe l'ordinamento crescente. Pertanto, `next` viene accantonato e inserito nell'array $U$ per far parte della _run_ successiva

La fase termina quando il min-heap $H$ si svuota, in questo momento, tutti gli $M$ elementi residenti in RAM si trovano nell'array $U$ e sono tutti minori dell'ultimo valore `min` estratto e gli $M$ elementi di $U$ vengono usati per costruire il nuovo min-heap $H$, l'array $U$ viene resettato e ha inizio una nuova fase per la _run_ successiva.

From here we have two important **observation**:
1. During the phase execution the minimum of $H$ in non-decreasing, and so it is also non-decreasing the output run.
	- $\Longrightarrow$  that implies the correctness of observation
2. The items stored in $H$ at the beginning of the phase are eventually output before the phase ends.
	- $\Longrightarrow$ that implies the that htis approach froms sorted runs longer than M, and thus its use makes the final algorithm not less efficient than Merge-Sort

> [!success] Theorem
> **Snow-Plow** build $O(n/M)$ sorted Runs, each longer than $M$ and actually of of length $2M$ on average.

We can demonstrate this in the following way:

> [!success] Proff
> We have $\tau$ that is the total number of items readed from the input during a phage.
> - A phase is over when $U$ is full of $M$ elements $\Longrightarrow$ so the $\tau$ elements read:
> 	- $M$ will be in $U$
> 	- $\tau - M$ will be in $H$
> - The final length of a run is determinated by the first $M$ starting elements + $\tau - M$ adding elemetnts entranti in $H$ during the scan: $$fianl_lenght = M + (\tau - M) = \tau$$
> - Assuming that $\mathbb(P)(\text{go to U or H}) = 1/2$ we will have that half items are going in $H$ and half in $U$: $$\tau/2 = M \Longrightarrow \tau = 2M$$

Using the Snow-flow for the formation of sorted runs Merge-Sort takes an average I/O complexity:
$$ O\left(\frac{n}{B} \log_{M/B} \frac{n}{2M}\right) \text{ I/O} $$
# References
- [[Pearls_of_Algorithm_Engineering_Ferragina_2023.pdf#page=63]]