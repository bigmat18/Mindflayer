**Data time:** 16:11 - 10-05-2025

**Status**: #note #master

**Tags:** [[Basi di Dati]] [[Operatori Insiemistici]] [[Algebra Relazionale]]

**Area**: [[Bachelor's Degree]]
# Intersezione
![[Screenshot 2023-11-26 at 18.15.16.png | 500]]

Possiamo derivare l'operatore intersezione usando gli operatori: $R(A,B), \:\: S(A, B)$, [[Prodotto]], [[Ridenominazione]], [[Selezione]], [[Proiezione]].

![[Screenshot 2023-11-26 at 18.16.29.png | 500]]

**Esempio**. Prove scritte in un concorso pubblico. I compiti sono anonimi e ad ognuno è associata una busta chiusa con il nome del candidato. Ciascun compito e la relativa busta vengono contrassegnati con uno stesso numero.

![[Screenshot 2023-11-26 at 18.21.43.png | 500]]

![[Screenshot 2023-11-26 at 18.21.56.png | 500]]

**Prodotto cartesiano a [[Chiavi nel Modello Relazionale|chiavi]] esterne**
![[Screenshot 2023-11-26 at 18.22.46.png]]
$$\pi_{Matr,Nome,Materia,Data,Voto}(\sigma_{matricola=s.matricola}(Esami \: \times \: \rho_s \: Stuedenti)$$
**Prodotto cartesiano, rappresentazione mediante albero**
![[Screenshot 2023-11-26 at 18.25.00.png]]

# References