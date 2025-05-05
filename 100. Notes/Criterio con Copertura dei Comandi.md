**Data time:** 14:10 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: [[Bachelor's Degree]]
# Criterio con Copertura dei Comandi

Si cercano valori per x e y che esercitino tutti i comandi
- { (x = 0, y = 0) } (non esercita tutti i comandi)
- { (x = 0, y = 0), (x = 2, y = 2) } (non esercita tutti i comandi)
- { (x = -2, y = 3), (x = 4, y = 0), (x = 0), (y = -5) } (esercita tutti i comandi)

![[Screenshot 2023-12-04 at 16.12.13.png]]

**Misura copertura dei comandi**. Misura di copertura = $\large\frac{numero \:\: di \:\: comandi \:\: esercitari}{numero \:\: di \:\: comandi \:\: totali}$

**Esempio**. 
Per avere una copertura totale servono almeno due casi di test uno con $y < 0$ ed uno con $y >= 0$. In particolare:
- { (x = 2, y = -2) } esercita i comandi lungo il cammino rosso ed ha una copertura di 8/9 = 89%
- { (x = 2, y = 0) } esercita i comandi lungo il cammino marrone ed ha una copertura di 6/9 = 66%
- { ( x = 2, y = -2),  (x = 2, y = 0) } ha una copertura di 9/9 = 100%

La copertura non è monotona rispetto alla dimensione dell’insieme di test:
{ (x=4, y=2) } ha una copertura più alta rispetto a { (x = 2, y = 0), (x = -2, y = 2)}

Al solito, cerchiamo di minimizzare il numero di test, a parità di copertura.
Ma non sempre vale la pena cercare a tutti i costi un insieme minimale che dia copertura al 100%.
# References