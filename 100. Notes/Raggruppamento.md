**Data time:** 16:11 - 10-05-2025

**Status**: #note #master

**Tags:** [[Basi di Dati]] [[Operatori Insiemistici]] [[Algebra Relazionale]]

**Area**: [[Bachelor's Degree]]
# Raggruppamento

Il raggruppamento si definisce come:
$${}_{\{A_i\}} \gamma_{f_i}(R)$$
Gli $A_i$ sono attributi di R e le $f_i$ sono espressioni che usano funzioni di aggregazione (min, max, count, sum, ...)

Il valore del raggruppamento è una relazione calcolata come segue:
- Si partizionano le ennuple di R mettendo nello stesso gruppo tutte le ennuple con valori uguali degli $A_i$.
- Si calcolano le espressioni $f_i$ per ogni gruppo.
- Per ogni gruppo si restituisce una sola ennupla con attributi i valori degli $A_i$ e delle espressioni $f_i$.
![[Screenshot 2023-12-06 at 13.30.00.png]]

**Esecuzione del raggruppamento**
Per ogni candidato: numero degli esami, voto minimo, massimo e medio:
$$\large{}_{\{Candidato\}}\gamma_{\{count(*), min(Voto),max(voto),avg(Voto)\}}(Esame)$$
![[Screenshot 2023-12-06 at 14.22.36.png]]

**Esempio** su due attributi.
![[Screenshot 2023-12-06 at 14.23.04.png]]

**Raggruppamento su [[Chiavi nel Modello Relazionale|chiave]] primaria**
![[Screenshot 2023-12-06 at 14.23.43.png]]

![[Screenshot 2023-12-06 at 14.24.15.png]]

# References