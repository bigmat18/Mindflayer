**Data time:** 01:49 - 28-11-2024

**Status**: #note #youngling 

**Tags:** [[Probabilità e (In)dipendenza]] [[Statistics]]

**Area**: 
# Indipendenza

L'indipendenza fra due eventi è la proprietà per cui la probabilità che avvenga uno dei due non influenza l'altro eventi. In altre parole
$$\mathbb{P}(A) = \mathbb{P}(A | B) \:\:\:\: \mathbb{P}(B) = \mathbb{P}(B|A)$$
SI può formalizzare dicendo che dati due eventi A e B, essi sono indipendenti quando:
$$\mathbb{P}(A \cap B) = \mathbb{P}(A) \cdot \mathbb{P}(B)$$
Ovviamente abbiamo che se $\mathbb{P}(A\cap B) \neq \varnothing$ i due eventi saranno dipendenti fra di loro.
- Se A e B sono indipendenti lo sono anche $A^C$ con $B$, $B^C$ con $A$ e $A^C$ con $B^C$ 
- Se $\mathbb{P}(A) = 0$ oppure $\mathbb{P}(A) = 1$ abbiamo che A è indipendente da qualsiasi evento
- Se due eventi sono disgiunti  ($A \cap B = \varnothing$) non possono essere indipendenti, almeno che uno dei due non sia trascurabile

Dati n eventi $A_1, \dots, A_n$ questi si dicono indipendenti se per ogni intero k con $2 \leq k \leq n$ e per ogni scelta di interi $\ \leq i_1 < i_2 < \dots \ i_k \leq n$ vale
$$\mathbb{P}(A_{i} \cap \dots \cap A_{i_{k}}) = \mathbb{P}(A_{i_{k}}) \dots \mathbb{P}(A_{i_k})$$ 
Inoltre se consideriamo un insieme $\Omega = \{a = (a_1, \dots, a_n) | a_i =0,1\} = \{0, 1\}^n$ dove per ogni $a = (a_1, \dots, a_n)$ definiamo:
$$\mathbb{P}(\{a\}) = p^{\#\{i:a_i=1\}}(i-p)^{\#\{i:a_i = 0\}}$$
lo spazio di probabilità e gli eventi $A_i = \{a \in \Omega : a_i = 1\}$ sono tutti indipedenti fra di loro.
# References