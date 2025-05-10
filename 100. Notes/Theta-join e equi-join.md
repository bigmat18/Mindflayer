**Data time:** 16:24 - 10-05-2025

**Status**: #note #master

**Tags:** [[Basi di Dati]] [[Operatori Insiemistici]] [[Algebra Relazionale]]

**Area**: [[Bachelor's Degree]]
# Theta-join e equi-join

Un join naturale su relazioni senza attributi in comune contiene sempre un numero di ennuple pari al prodotto delle cardinalità degli operandi (le ennuple sono tutte combinabili).

Il prodotto cartesiano concatena tutle non necessariamente correlate dal punto di vista semantico.

![[Screenshot 2023-11-26 at 21.17.23.png]]

Il prodotto cartesiano è più utile se fatto seguire da una selezione. Prodotto cartesiano seguito dalla selezione che mantiene solo le tuple con valori uguali sull’attributo: Reparto di Impiegati e Codice di Reparti.

Il prodotto cartesiano, in pratica, ha senso solo se seguito da selezione:$\sigma_{condizione}(R_1 \bowtie R_2)$ 

L'operazione viene chiamata **theta-join** e può essere sintatticamente indicata con $R_1 \bowtie_{condizione} R_2$ 

Perché "theta-join"? La condizione $C$ è spesso una congiunzione ($\land$) di atomi di confronto di $A_1 \upvartheta A_2$ dove $\upvartheta$ è uno degli operatori di contronto ($=, >, <, \dots$) Se l'operatore è sempre l'uguaglianza ($=$) allora si parla di **equi-join**.

# References