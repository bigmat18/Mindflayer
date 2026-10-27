---
Data: 2026-10-25T22:54:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Sorting Atomic Items]]"
Area:
  - "[[Master's Degree.base]]"
---
# Pivoting Selection in QS

In [[Quick-Sort]] the selection of the pivot is crucial to get balanced partitions, reduce the number of recursive calls and achieve optimal $O(n\log n)$ **time complexity**.

We can use differents approach that will bring to different behavior:
1. In the basic implementation we choose the first element in the input array but it is easy to instantiate the input array in order to induce unbalanced partitions: we can just take S to be an increasing or decreasing ordered sequence of items. 
2. one alternative way to prevent nbalanced partitions: is to select **pivot randomly** among the items in $S[i,j]$. ut this makes the behavior of the algorithm unpredictable in advance and dependant on the random selection of the pivot. We can show that the **expected time complexity** is the optimal $O(n log2 n)$, with a hidden constant smaller than 2

> [!success] Theorem
> The **random selection** of the pivot drives QUICKSORT to compare no more than $2n \ln n$ items, in expectation.

> [!success] Proof
> We wish to compute the number of comparisons executed by `PARTITION` over the input sequence $S$. First of all:
> - Let $X_{u,v}$ be the random binary variable that indicates whether $S[u]$ and $S[v]$ are compared by `PARTITION`:
>     $$X_{u,v} = \begin{cases} 1 & \text{se } S[u] \text{ e } S[v] \text{ vengono confrontati durante la procedura } \text{PARTITION} \\ 0 & \text{altrimenti} \end{cases}$$
> - Sia $p_{u,v} = E[X_{u,v}]$ la probabilità che il confronto tra $S[u]$ e $S[v]$ avvenga effettivamente.
> The expected number of comparisons executed by Quicksort can then be computed as:
> $$E\left[\sum_{u<v} X_{u,v}\right] = \sum_{u=1}^{n} \sum_{v=u+1}^{n} E[X_{u,v}] = \sum_{u=1}^{n} \sum_{v=u+1}^{n} p_{u,v}$$
> 
> To estimate $p_{u,v}$ we concentrate on the random choice of the pivot $S[r]$, because two items are compared by `PARTITION` only if one of them is the pivot. We can distinguish 3 cases:
> 
> - **Case 1:** $S[r]$ is smaller or greater than both $S[u]$ and $S[v]$. In this case, $S[u]$ and $S[v]$ will be compared with $S[r]$ but never with each other.
> - **Case 2:** $S[r]$ is either $S[u]$ or $S[v]$. In this case, the two elements are going to be compared just one time (a good event: $1$).
> - **Case 3:** $S[r]$ is between $S[u]$ and $S[v]$. The two elements are going to be split into two sub-arrays, so this is a bad event: $0$ (they will never be compared). It can also be seen as the complement of Case 2.
> The most important case is Case 3. Since the pivot chosen will be random, the probability $p_{u,v}$ that a comparison happens will be the ratio between the positive cases (casi favorevoli) and total cases:
> $$p_{u,v} = \frac{\text{Casi Favorevoli (Caso 2)}}{\text{Pivot Decisivi Totali (Caso 2 + Caso 3)}} = \frac{2}{v' - u' + 1}$$
> 
> This formula can also be derived with the following idea: to determine the probability of the comparison we have to determine the probability of choosing the pivot as $S[u]$ or $S[v]$:
> - There are two cases where the comparison is true.
> - There are $b$ cases where the comparison is false.
> - If we consider $S'$ as the sorted array $S$, and $S[u]$ mapped to $S'[u']$ and $S[v]$ mapped to $S'[v']$, the value of $b$ is $v' - u' - 1$, and so:
>$$p_{u,v} = \frac{2}{b+2} = \frac{2}{v' - u' + 1}$$
>
>From here we can develop the sum value:
> $$E[\text{Confronti Totali}] = \sum_{u'=1}^{n} \sum_{v'=u'+1}^{n} \frac{2}{v' - u' + 1}$$
> 
> Fissato l'indice $u'$, definiamo la nuova variabile di distanza $k = v' - u' + 1$:
> - Quando $v' = u' + 1 \implies k = (u'+1) - u' + 1 = 2$.
> - Quando $v' = n \implies k = n - u' + 1$.
>
>Sostituendo $k$ nella sommatoria interna per un generico $u'$:
>$$\sum_{v'=u'+1}^{n} \frac{2}{v' - u' + 1} = \sum_{k=2}^{n - u' + 1} \frac{2}{k}$$
> 
> Raccogliamo la costante $2$ fuori dalle sommatorie:
> $$E[\text{Confronti Totali}] = \sum_{u'=1}^{n} \left( \sum_{k=2}^{n - u' + 1} \frac{2}{k} \right) = 2 \sum_{u'=1}^{n} \sum_{k=2}^{n - u' + 1} \frac{1}{k}$$
> Notiamo che per qualsiasi $u' \ge 1$, l'estremo superiore della somma interna soddisfa $n - u' + 1 \le n$. Possiamo quindi maggiorare la sommatoria estendendo l'estremo superiore a $n$ per tutti i termini $u'$:
> $$\sum_{k=2}^{n - u' + 1} \frac{1}{k} \le \sum_{k=2}^{n} \frac{1}{k}$$
> Sostituendo questa disuguaglianza:
> $$E[\text{Confronti Totali}] \le 2 \sum_{u'=1}^{n} \left( \sum_{k=2}^{n} \frac{1}{k} \right)$$
> Poiché la sommatoria interna $\sum_{k=2}^{n} \frac{1}{k}$ è ora completamente indipendente dall'indice $u'$, possiamo portarla fuori dalla sommatoria esterna:
>$$2 \sum_{u'=1}^{n} \left( \sum_{k=2}^{n} \frac{1}{k} \right) = 2 \cdot \left( \sum_{u'=1}^{n} 1 \right) \cdot \left( \sum_{k=2}^{n} \frac{1}{k} \right) = 2n \sum_{k=2}^{n} \frac{1}{k}$$
> La sommatoria $\sum_{k=2}^{n} \frac{1}{k}$ corrisponde al numero armonico $H_n$ privo del primo termine ($H_n - 1$). Dalle proprietà dell'analisi matematica (maggiorazione tramite l'integrale $\int_1^n \frac{1}{x} \, dx = \ln n$):
> $$\sum_{k=2}^{n} \frac{1}{k} = H_n - 1 \le \ln n$$
> Sostituendo il logaritmo naturale $\ln n$:
> $$E[\text{Confronti Totali}] \le 2n \ln n$$
>From here I can convert everthing in base 2:
>$$2n \ln n = 2 \cdot (\ln 2) \cdot n \log_2 n \approx 2 \cdot (0.69315) \cdot n \log_2 n \approx 1.3863 \, n \log_2 n$$

### Dual Pivoting
The next question is how we can “enforce” the expected behavior. The natural answer is to sample more than one pivot. Typically, three pivots are randomly sampled from S and the median is taken, thus requiring just two comparisons. Taking more than three pivots makes **the selection of a “good one” more robus**, it is describer in the following thorem

> [!success] Theorem
> if QUICKSORT partitions around the median of 2s + 1 randomly selected items, it sorts n distinct items in
$$ \frac{2nH_n}{H_{2s+2} - H_{s+1}} + O(s)$$ 
expected comparisons, where $H_z$ is the z-th harmonic number $\sum_{i=1}^{z}\frac{1}{i}$

By increasing s, we can push the expected number of comparisons close to $n \log n + O(n)$, however the selection of the median incurs a higher cost. Randomization helps in simplifying the selection and still guarantees $O(s)$ expected time performance. So let's see a randomize versione for [[Quick-Sort#Three-Way Partitioning]]

![[Pasted image 20261027233102.png|524]]

The algorithm above is randomized and selects the item of the unordered $S$ having rank $k$. 

> [!NOTE] Observation
> It is interesting to see that the algorithmic scheme mimics the one used in the **Partitioning phase** of QUICKSORT: here the selected item $S[r]$ plays the same role of the pivot in QUICKSORT, because it is used to partition the input sequence S in **three parts** consisting of items smaller/equal/larger than $S[r]$.
> 
> But unlike QUICKSORT, RANDSELECT recurses only in one of these three parts, namely the one containing the k-th ranked item, which can be determined by just looking at the sizes of those parts, as done in Steps 6 and 8

![[Pasted image 20261027235203.png|307]]

There are two specific issues that deserve a comment:
1. we do not need to recurse on S= because it consists of items all equal to S[r], so the k-th ranked item has value S[r].
2. if recursion occurs on $S_>$, we need to update the searched rank k because we are dropping from the original sequence the items belong- ing to the set $S_< ∪ S_=$.


> [!success] Theorem
> Selecting the k-th ranked item in an unordered sequence of size n takes O(n) expected time in the RAM model, and O(n/B) expected I/Os in the two-level memory model.

> [!success] Proof
> Let's call "good selection" a pivot $S[r]$ that induces a partition in which $n_<$ and $n_>$ are not larger that $2n/3$. 
> - To do that the pivot should have a value with rango in $[n/3, 2n/3]$.
> - This occurs with a probability of $1/3$
> $$
> P(n/3 \leq n_< \leq 2n/3) = \frac{\text{good case}}{\text{all cases}} = \frac{n/3}{n} = \frac{1}{3}
> $$
> Sia $\hat{T}(n)$ il tempo atteso per eseguire `RANDSELECT` su un input di dimensione $n$ nel modello RAM. Possiamo impostare la seguente disuguaglianza di ricorrenza:
> $$\hat{T}(n) = O(n) + \frac{1}{3} \cdot \hat{T}\left(\frac{2n}{3}\right) + \frac{2}{3} \cdot \hat{T}(n-1) \le O(n) + \frac{1}{3} \cdot \hat{T}\left(\frac{2n}{3}\right) + \frac{2}{3} \cdot \hat{T}(n)$$
> 
>where we have:
> - $O(n)$: È il tempo deterministico necessario per eseguire i passaggi 2-5 (scansione dell'array per costruire $S_<, S_>$ e $S_=$).
> - $\frac{1}{3} \cdot \hat{T}\left(\frac{2n}{3}\right)$: Con probabilità $\ge 1/3$ la selezione è "buona", quindi la dimensione del sotto-problema ricorsivo si riduce ad almeno $\frac{2}{3}n$.
> - $\frac{2}{3} \cdot \hat{T}(n)$: Con probabilità $\le 2/3$ la selezione è "cattiva". Usiamo un limite superiore conservativo ipotizzando che la dimensione del sotto-problema non si sia ridotta affatto e rimanga $n$.
> 
> Notiamo che il termine sconosciuto $\hat{T}(n)$ compare sia a sinistra sia a destra dell'equazione. Possiamo isolarlo algebricamente:
> 1. Portiamo $\frac{2}{3} \hat{T}(n)$ al membro di sinistra sottraendolo da entrambi i lati:$$\hat{T}(n) - \frac{2}{3} \hat{T}(n) \le O(n) + \frac{1}{3} \cdot \hat{T}\left(\frac{2n}{3}\right)$$
> 2. Semplifichiamo il membro di sinistra: $$\frac{1}{3} \hat{T}(n) \le O(n) + \frac{1}{3} \cdot \hat{T}\left(\frac{2n}{3}\right)$$
> 3. Moltiplichiamo entrambi i membri per $3$: $$\hat{T}(n) \le 3 \cdot O(n) + \hat{T}\left(\frac{2n}{3}\right) \implies \hat{T}(n) \le O(n) + \hat{T}\left(\frac{2n}{3}\right)$$
> 4. Sviluppiamo la ricorrenza unrollando la serie geometrica: $$\hat{T}(n) \le c \cdot n + c \cdot \left(\frac{2}{3}n\right) + c \cdot \left(\frac{2}{3}\right)^2 n + c \cdot \left(\frac{2}{3}\right)^3 n + \dots$$ $$\hat{T}(n) \le c \cdot n \sum_{i=0}^{\infty} \left(\frac{2}{3}\right)^i$$
> 5. Poiché la serie geometrica converge a $\frac{1}{1 - 2/3} = 3$: $$\hat{T}(n) \le 3 \cdot c \cdot n = O(n)$$

# References