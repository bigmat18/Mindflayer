**Data time:** 16:11 - 10-05-2025

**Status**: #note #master

**Tags:** [[Basi di Dati]] [[Operatori Insiemistici]] [[Algebra Relazionale]]

**Area**: [[Bachelor's Degree]]
# Proiezione

Operatore monadico, produce un risultato che, ha parte degli attributi dell'operando, contiene ennuple cui contribuiscono tutte le ennuple dell'operando ristrette agli attributi nella lista.
Sintassi: $\pi_{ListaAttributi}(Operando)$

**Esempio**. Matricola e cognome di tutti gli impiegati.
![[Screenshot 2023-11-26 at 17.02.22.png]]
**Proiezione**: $\pi_{Nome, Matricola}(Studenti)$

**Esempio**. Con duplicati
![[Screenshot 2023-11-26 at 17.20.01.png]]

Una proiezione contiene al più tante ennuple quante l'operando, può ovviamente contenerle meno.
Se $X$ è una [[Chiavi nel Modello Relazionale|superchiave]] di $R$, allora $\pi_X(R)$ contiene esattamente tante ennuple quante $R$.
Mentre se $X$ non è una superchiave, potrebbe esistere valori ripetuti su queli attributi, che quindi vengono rappresentati una sola volta.
# References