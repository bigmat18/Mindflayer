**Data time:** 14:22 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: [[Bachelor's Degree]]
# Criterio con Copertura delle Decisioni

Con (x=2,y=-1) si esercitano tutti i comandi ma… bisogna avere casi di test che esercitino entrambi i rami di ogni condizione. Per avere una copertura delle decisioni occorre avere almeno due casi di test: uno y<0 e uno y>=0. (Per coprire tutte le frecce).

**Misura di copertura** = $\large\frac{numero \: di \: archi \: esercitati}{numero \: di \: archi \: totali}$

Con (x = 2, y = -1) copertura di $\frac{9}{11}$ delle decisioni.

### Condizioni composte
```java
if (x > 1 || y == 0) { comando1 }
else { comando2 }
```
- Il test { x=0, y=0 } e { x = 0, y = 1 } garantisce la piena copertura delle decisioni ma non esercita tutti i valori di verità della prima condizione.
- Il test { x=2, y=2 } e { x=0, y=0 } esercita i valori di verità delle due condizioni (ma non tutte le decisioni)
- Il test { x= 2, y=0 }, {x=0, y=1 } esercita tutti i valori di verità delle due condizioni e tutte le decisioni.

### Copertura delle condizioni semplici
Un insieme di test T per un programma P copre tutte le condizioni semplici (basic condition) di P se, per ogni condizione semplice CS in P, T contiene un test in cui CS vale true e un test in cui CS vale false.$$Copertura\:delle\:basic\:condition\:=\: \large\frac{n. \:di\: valori\: assunti\: dalle\: basic\:condition}{2 * n.\: di\: basic\: conditions}$$
**Condizioni composte**
Nel esempio di prima
- Il test { x=0, y=0 } e { x = 0, y = 1 } ha copertura delle condizioni semplici: $\frac{3}{4}$.
- Il test { x=2, y=2 } e { x=0, y=0 } ha copertura delle condizioni semplici: $\frac{4}{4}$.
- Il test { x= 2, y=0 }, {x=0, y=1 } ha copertura delle condizioni semplici: $\frac{4}{4}$.

Prendiamo ora per esempio il codice:
```java
if (x > 1 && y == 0 && z > 3) { comando1 }
else { comando2 }
```
La multiple condition coverge richiede di testare le possibili combinazioni ($2^n$ con $n$ condizioni semplici).
Nell'esempio sarebbero $2^3$ casi ma (semantica java di &&) si può ridurre da 8 a 4 perché: vero vero vero; vero vero falso; vero falso _ ; false, _ , _
# References