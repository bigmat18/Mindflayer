**Data time:** 16:11 - 10-05-2025

**Status**: #note #master

**Tags:** [[Basi di Dati]] [[Operatori Insiemistici]] [[Algebra Relazionale]]

**Area**: [[Bachelor's Degree]]
# Giunzione

Combinando selezione e proiezione, possiamo estrarre informazioni da una relazione, ma non possiamo però correlare informazioni presenti in relazioni diverse.

il join è l'operatore più interessante dell'algebra relazionale poiché permette di correlare dati in relazioni diverse.

### [[Join naturale]]
### Cardinalità delle join
Prendiamo $R_1(A,B), \: R_2(B,C)$, Il join di $R_1$ e $R_2$ contiene un numero di ennuple compreso fra zero ed il prodotto di $|R_1|$ e $|R_2|$:
$$ 0 \:\: \leq \:\: |R_1 \bowtie R_2| \:\: |R_1| \times |R_2|$$
Se la join fra $R_1$ ed $R_2$ è completo, allora contiene un numero di ennuple almeno uguale al massimo fra $|R_1|$ e $R_2$. Se la join coinvolge una chiave B  di $R_2$ allora il numero di ennuple è compreso fra zero e $|R_1|$:
$$0 \:\: \leq |R_1 \bowtie R_2| \:\: \leq |R_1|$$
Se il join coinvolge una chiave $B$ di $R_2$ e un vincolo di integrità referenziale tra attributi di $R_1$ ($B$ in $R_1$) e la chiave di $R_2$, allora il numero di ennuple è pari a $|R_1|$:
$$|R_1 \bowtie R_2| = |R_1|$$
Una difficoltà della join:
![[Screenshot 2023-11-26 at 21.06.46.png | 500]]

### [[Join esterno]]
### Join e [[Proiezione]]

![[Screenshot 2023-11-26 at 21.15.06.png | 400]]

![[Screenshot 2023-11-26 at 21.15.38.png | 400]]

### [[Theta-join e equi-join]]
### [[Join]]
### [[Self join]] 


# References