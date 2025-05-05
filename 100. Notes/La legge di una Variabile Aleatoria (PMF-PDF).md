**Data time:** 13:06 - 28-11-2024

**Status**: #note #youngling 

**Tags:** [[Variabili Aleatore]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# La legge di una Variabile Aleatoria (PMF-PDF)

Una variabile aleatoria è una funzione $X: \Omega \to \mathbb{R}$ che associa ad ogni valore dello spazio campionario un numero reale. questo permette di rappresentare numericamente il risultato di un fenomeno aleatorio.

Data una $X: \Omega \to \mathbb{R}$ la funzione di insieme $\mathbb{P}_X$ si dice **legge di probabilità**. E definisce come la probabilità si distribuisce sui possibili valori assunti da X. Essa si definisce diversamente per V.A. discrete e continue (con densità)

![Variabili aleatorie - distribuzione di probabilità | 200](https://www.edutecnica.it/calcolo/casuali/1a.png)
##### Variabili aleatorie discrete (PMF)
Una V.A. si dice discreta quando la sua immagine $X(\Omega) \subset \mathbb{R}$ è un sottoinsieme al più numerabile di $\mathbb{R}$ 
$$\mathbb{P}_X(A) = \mathbb{P}(X \in A) = \sum_{x_i \in A}p_X(x_i)$$
##### Variabili aleatorie con densità (PDF)
Una V.A. è detta con densità se la legge di probabilità è definita da una densità $f$, cioè se esiste una densità di probabilità $f$ tale che valga
$$\mathbb{P}_X(A) = \mathbb{P}\{X \in A\} = \int_A f(x) dx$$
# References