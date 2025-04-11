**Data time:** 14:05 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: 
# Partizione dei dati in ingresso in classi di equivalenza

Questo è un metodo in cui il dominio dei dati di ingresso è ripartito in **classi di equivalenza**. Due valori d'ingresso appartengono alla stessa classe di equivalenza se, in base ai requisiti, dovrebbero produrre lo stesso comportamento del programma (non necessariamente stesso output).

**Esempio**
Abbiamo un metodo ```int calcolaTasse(int reddito)```. Il test obligation è un caso di test per aliquota, i casi di test che soddisfano le test obligations sono: <10.000, 2300, _>, <20.000, 4800, _>, ...
![[Screenshot 2023-12-04 at 11.35.42.png]]

Il criterio è economicamente valido solo per quei programmi per cui il numero dei possibili comportamenti è sensibilmente inferiore alle possibili confiugurazione d'ingresso. Per come sono costruite le classi, i risultati attesi dal test sono noti e quindi non si pone il problema dell'oracolo.

Il criterio è basato su un'affermazione generalmente plausibile, ma non vera in assoluto, la deduzione che il corretto funzionamento sul valore rappresentate implichi la correttezza su tutta la classe di equivalenza dipende dalla realizzazione del programma e non è verificabile sulla base delle sole specifiche funzionali. 
#### Valori limite (di frontiera)
Metodo basato sull'individuazione di valori estremi. 
- Estremi delle classi di equivalenza definite in base all'eguaglianza del comportamento del programma.
- Estremi in base a considerazioni inerenti il tipo dei valori d'ingresso (per esempio se interi considerare 0 e 1).

**Esempio**.
Abbiamo il metodo ```int calcolaTasse(int reddito)```. Il test obligation è provare tutti gli intorni degli estremi degli intervalli. I casi di test che soddisfano le test obbligations sono: <14.990, 3.447,7, _> <15.000, 3450, _>, <15.010, 3452,7 ,_> ….
(Per questa specifica è poso significativo questo criterio: sui punti di frontiera non è derivabile ma è comunque continua)

**Esempio** più significativo.
Frontiera punto di discontinuità. Il metodo è sempre ```int calcolaSconto(int spesa)```
![[Screenshot 2023-12-04 at 11.43.08.png]]
I casi di test che soddisfano le test obligations: <48.99 , 0, _>, <49.00, 7.35, _>, <49.01 , 7.3515 ,_> …

Questo criterio ricorda i controlli sui valori limite tradizionali in altre discipline ingegneristiche per le quali è vera la proprietà del comportamento continuo (ad esempio in meccanica una parte provata per un certo caricoresiste con certezza a tutti i carichi inferiori).

In realtà si guardano i valori limite perché spesso è nell'intorno dei valori limite che si nascono difetti del codice.

**I casi non validi**
Per ogni input si definiscono anche i casi non validi (chedevono generare un errore): età inferiori a 20 o superiori a 120 per la laurea, reddito negativo per il calcolo delle aliquote, spesa negativa per il calcolo dello sconto.
# References