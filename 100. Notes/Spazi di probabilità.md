**Data time:** 17:32 - 26-11-2024

**Status**: #note #youngling 

**Tags:** [[Probabilità e (In)dipendenza]] [[Statistics]]

**Area**: 
# Spazi di probabilità

Un insieme di tutti i dati possibili possono essere rappresentati all'interno di uno insieme astratto $\Omega$ detto **spazio campionario**. Un sottoinsieme di $\Omega$ è un **evento**. Consideriamo $\mathcal{P}(\Omega)$ con l'insieme di tutti i possibili sottoinsiemi di $\Omega$.

##### Algebra di parti
Dato un $\mathcal{F} \subset \mathcal{P}(\Omega)$ esso è un **algebra di parti** se:
- contiene l'insieme vuoto $\varnothing$  e l'insieme $\Omega$
- è chiuso per complementari, ovvero se $A \in \mathcal{P}$ allora $A^c \in \mathcal{F}$
- è chiuso per unioni (finite), ovvero se $A_1, ..., A_n \in \mathcal{P}$ allora $A_1 \cup ... \cup A_n \in \mathcal{F}$ 
##### $\sigma$-algebra
Abbiamo che se $\mathcal{P}$ è un **algebra di parti** può essere anche una $\sigma$-algebra se
- è chiusa per unioni numerabili, ovvero se esiste una successione $A_1, ..., A_n, \dots \in \mathcal{P}$ allora $A_1 \cup ... \cup A_n \in \mathcal{P}$

Questa definizione permette di eseguire tutte le operazioni insiemistiche anche co $\Omega$ infinito
##### Misura di probabilità
Il grado di fiducia, o probabilità che un evento si realizzi viene rappresentata da un valore fra 0 e 1. E se due eventi sono disgiunti (intersezione vuota) la probabilità che uno dei due si realizzi è la somma fra i due           (**proprietà additiva**).
$$\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) \:\:\: se \:\:\: A \cap B = \varnothing$$
La **misura di probabilità** è una funzione $\mathbb{P}: \mathcal{P} \to [0, 1]$ t.c.:
- L'evento certo ha probabilità unitaria 
- Dalla proprietà additiva abbiamo che se abbiamo una successione di eventi disgiunti
$$\mathbb{P}\left( \bigcup_{n=1}^{+\infty}A_{n} \right) = \sum_{{n=1}}^{+\infty}\mathbb{P}(A_{n})$$
Si dice **trascurabile** un evento con probabilità 0, si dice **quasi certo** un evento con probabilità 1

# References