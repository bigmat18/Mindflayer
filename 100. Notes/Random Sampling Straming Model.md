---
Data: 2026-10-28T17:40:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Random Sapling]]"
Area:
  - "[[Master's Degree.base]]"
---
# Random Sampling Straming Model

We change the basic case in [[Random Sampling Disk Model]] in a situation where $S$ is flowing through a channel and the input size $n$ is known and large. This bring to:
- **Nessun pre-processing sui dati**: Non è possibile generare posizioni e ordinarle in anticipo come nella Sezione 3.11.
- **Scansione singola (Single-pass)**: Ogni elemento della sequenza $S[1 \dots n]$ viene esaminato una sola volta.
- **Decisioni immediate e irrevocabili**: Quando un elemento arriva, l'algoritmo deve decidere all'istante se includerlo o meno nel campione. Un elemento scartato non potrà mai più essere recuperato in futuro.
- **Assenza di Resampling**: A differenza degli algoritmo in [[Random Sampling Disk Model]], l'approccio streaming garantisce che ogni elemento venga preso in considerazione una sola volta, eliminando la generazione di duplicati da scartare.

To design this new approach let's start with the simple case $m=1$. We want analyse the probability to extract a single element uniformly from the flow.

L'algoritmo assegna a ciascun elemento $S[j]$ una probabilità di selezione definita come: 
$$
\mathcal{P}(j) = \frac{1}{n - j + 1}
$$
- Per il primo elemento $S$, la probabilità è $\mathcal{P}(1) = \frac{1}{n}$.
- Per il secondo elemento $S$, è $\mathcal{P}(2) = \frac{1}{n-1}$.
- Per l'ultimo elemento $S[n]$, è $\mathcal{P}(n) = \frac{1}{1} = 1$.
Non appena un elemento viene selezionato, l'algoritmo si arresta.

To sample an element $S[j]$ we need that any element before $j-1$ has been selected and $S[i]$ is selected.
$$
\mathcal{P}(\text{Campionare } S[j]) = \mathcal{P}(\text{Non campionare } S[1 \dots j-1]) \times \mathcal{P}(\text{Selezionare } S[j])
$$
Per induzione, la probabilità di non aver selezionato nessuno dei primi $j-1$ elementi è $1 - \frac{j-1}{n} = \frac{n - j + 1}{n}$. Moltiplicando le due probabilità otteniamo:
$$
\mathcal{P}(\text{Campionare } S[j]) = \left(\frac{n - j + 1}{n}\right) \times \left(\frac{1}{n - j + 1}\right) = \frac{1}{n}
$$
That **demonstrate the uniformly** of this approach.

Now let's look an algorithm that extend this idea:

![[Pasted image 20261028182619.png|524]]

The main difference in the algorithm above with the previus version is that the probability is:
$$
P(j) = \frac{m-s}{n-j+1}
$$
where:
- $s$ is the number of items already selected before $S[j]$
- $n$ is the length of the sequence
- $j$ is the current iteration
- $m$ is the number of sampling

**Note** two important consideration with this new formalization:
- that if we have already got all samples, it is $s=m$ and thus $P(j) = 0$, which means that the algorithm does not generate more than $m$ samples. 
- In other hand, the algorithm cannot generate less than $m$ samples because if we have $y$ samples with $y < m$ the last $m-y$ items of $S$ would have probability 1 of being selected and thus they would be included in the final sample.

This probability is obtained by divide:
1. the number that represent how many such combinations include $S[j]$:
2. and the number of combinations that eight include or do not include $S[j]$

$$
\mathcal{P}(j) = \frac{\binom{n-j}{m-s-1}}{\binom{n-j+1}{m-s}} = \frac{m-s}{n-j+1}
$$
# References