**Data time:** 00:45 - 28-11-2024

**Status**: #note #youngling 

**Tags:** [[Probabilità e (In)dipendenza]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Probabilità condizionale e formula di Bayes

##### Probabilità condizionale
In uno spazio di probabilità si chiama **probabilità condizionale** di A rispetto a B il numero:
$$\mathbb{P}(A | B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}$$
Indica la probabilità che si verifichi A nell'ipotesi che si verifichi B. A è l'evento **condizionato** e B è l'evento **condizionante**. Dati A e B non trascurabili:
$$\mathbb{P}(A \cap B) = \mathbb{P}(A | B) \cdot \mathbb{P}(B) = \mathbb{P}(B | A) \cdot \mathbb{P}(A)$$
##### Formula probabilità totale o fattorizzazione 
Una partizione di $\Omega$ è una collezione di n eventi $B_1, \dots, B_n$ a due a due disgiunti t.c. $B_1 \cup \dots \cup B_n = \Omega$. Un **sistema di alternative** è una partizione di $\Omega$ in eventi non trascurabili.

Sia $B_1, \dots, B_n$ un sistema di alternative. Allora per un qualcunque evento A vale:
$$\mathbb{P}(A) = \sum_{i=1}^n \mathbb{P}(A|B_{i})\mathbb{P}(B)$$
Vuol dire che, se consideriamo un evento A che può dipendere da una serie di eventi B, la prob. di A dipende dalle singole probabilità $\mathbb{P}(A|B)$ moltiplicate la prob che avvenga B.
##### Formula di Bayes
Siano A e B due eventi non trascurabili. Vale la seguente formula
$$\mathbb{P}(B|A) = \frac{\mathbb{P}(A|B)\mathbb{P}(B)}{\mathbb{P}(A)}$$
Serve per calcolare la probailità di B nel caso in cui sia avvenuto A.
# References