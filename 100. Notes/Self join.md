**Data time:** 16:23 - 10-05-2025

**Status**: #note #master

**Tags:** [[Basi di Dati]] [[Operatori Insiemistici]] [[Algebra Relazionale]]

**Area**: [[Bachelor's Degree]]
# Self join

Supponiamo di considerare la seguente relazione e di volere ottenere una relazione Nonno-Nipote.
![[Screenshot 2023-11-26 at 21.25.28.png]]
È ovvio che in questo caso abbiamo bisogno di utilizzare due volte la stessa tabella. Tuttavia Genitore $\bowtie$ Genitore = Genitore, poiché tutti gli attributi coincidono. 
In questo caso è utile effettuare una ridenomianzione:
$$\rho_{Nonno, Genitore\: \leftarrow \: Genitore, Figlio}(Genitore)$$
A questo punto effettuando una natural join con la tabelle Genitore, si ottiene l'informazione cercata.
![[Screenshot 2023-11-26 at 21.28.30.png]]
Eventualmente si può effettuare una proiezione.![[Screenshot 2023-11-26 at 21.29.47.png]]

**Esempio**. Join studenti ed esami
![[Screenshot 2023-12-06 at 12.34.47.png]]

Problemi: Matricola, Nome, cognome, voto degli studenti: 
- con (almeno un) voto maggiore di 28 (quantificatore esistenziale)
- non hanno mai ottenuto un voto maggiore di 28 (differenza)
- Nomi degli studenti che hanno ottenuto solo voti maggiore di 28 (quantificatore universale)

Esempio di trasformazione su quantificatore esistenziale.
Matricola, Nome, cognome, materia, data, voto degli studenti con voto maggiore di 28.
![[Screenshot 2023-12-06 at 12.35.42.png]]

**Non distribuititivà della proiezione rispetto alla differenza**
In generale:  $\pi_A(R_1 - R_2) <> \pi_A(R_1) - \pi_A(R_2)$
Se $R_1$ e $R_2$ sono definite su AB, e contengono tuple uguali su A e diverse su B.
![[Screenshot 2023-12-06 at 12.47.02.png]]
Dipende da chi è A.
Se $R_1$ e $R_2$ sono definite su AB e contengono tuple uguali su A e diverse su B, NO.
# References