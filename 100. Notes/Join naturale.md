**Data time:** 16:25 - 10-05-2025

**Status**: #note #master

**Tags:** [[Basi di Dati]] [[Operatori Insiemistici]] [[Algebra Relazionale]]

**Area**: [[Bachelor's Degree]]
# Join naturale


![[Screenshot 2023-11-26 at 18.26.15.png]]

Operatore binario (generalizzabile) che correla dati in relazioni diverse, sulla base di valori uguali in attributi con lo stesso nome.
- Produce un risultato: sull'unione degli attributi degli operandi con ennuple che sono ottenute combinando le ennuple degli operandi con valori uguali sugli attributi in comune.
- $R_1(X_1), R_2(X_2)$
- $R_1 \bowtie R_2$ è una relazione su $X_1 \cup X_2$.$$ R_1 \bowtie R_2 = \{ \: t \: su \: X_1 \cup X_2 \:\: \textbar \:\: t_1 \in R_1 \: e \: t_2 \in R_2 \: con \: T[X_1] = t_1\: e\: T[X_2] = t_2 \}$$
**Join naturale e attributi uguali.**
![[Screenshot 2023-11-26 at 18.30.57.png]]

**Join non completo.**![[Screenshot 2023-11-26 at 18.31.46.png]]

**Join vuoto.**
![[Screenshot 2023-11-26 at 18.32.01.png]]

**Join completo, con $n \times m$ ennuple**
![[Screenshot 2023-11-26 at 18.32.57.png]]


# References