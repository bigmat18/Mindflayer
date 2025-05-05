**Data time:** 14:05 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: [[Bachelor's Degree]]
# Test basato su catalogo

Nel tempo un organizzazione può essersi costruita un'esperienza nel definire casi di test. Collezionare questa esperienza in un catalogo può rendere più veloce il processo e automatizzare alcune decisioni riducendo l'errore umano.
I cataloghi catturano l'esperienza di coloro che definiscono i test elencando tutti i casi che devono essere considerati per ciascun possibile tipo di variabile. 

**Esempio** di voce nel catalogo.
Assumiamo che una funzione usi una variabile in cui valore deve appartenere ad un intervallo di interi, il catalogo potrebbe indicare i casi seguenti come rilevanti.
1. L'elemento che precede immediatamente il lower bound dell'intervallo.
2. Il lower bound.
3. l'upper bound
4. Un elemento non confinabile entro l'intervallo.
5. L'elemento su sussegue immediatamente l'upper bound.

Di fatto si stanno considerando: l'intervallo in cui è definita la funzione come se fosse un'unica classe di equivalenza, la sua frontiera, i valori non validi.
# References