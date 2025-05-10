**Data time:** 16:28 - 10-05-2025

**Status**: #note #master

**Tags:** [[Basi di Dati]] [[Algebra Relazionale]]

**Area**: [[Bachelor's Degree]]
# Trasformazioni Algebriche

Basate su regole di equivalenza fra espressioni algebriche. Consentono di scegliere diversi origini di [[Giunzione|join]] e di anticipare [[Proiezione]] e [[Restrizione]]. Alcuni esempi con relazioni R(A, B, C, D):
![[Screenshot 2023-12-06 at 14.25.31.png]]

**Atomizzazione delle sezioni**
$$\sigma_{F1 \land F2}(E) = \sigma_{F1}(\sigma_{F2}(E))$$
Una selezione congiuntiva può essere sostituita da una cascata si selezioni atomiche.

**Idempotenza delle Proiezioni**
Una proiezione può essere trasformata in una cascata di proiezioni che eliminano i vari attributi in fasi diverse.
$$\pi_X(E) = \pi_X(\pi_{XY}(E))$$
se E è definita su un insieme di attributi che contiene Y oltre che X. Non ha effetto sull’efficienza. Ha effetto sulla leggibilità della query.

**Anticipazione della selezione rispetto al Join**
**Pushing selection down**
$$\sigma_F(E_1 \bowtie E_2) = E_1 \bowtie \sigma_F (E_2)$$
se F fa riferimento solo ad attributi di E2 . Aumenta l’efficienza della query perché la selezione riduce il numero delle righe di E2 prima del join.

**Pushing projection down**
$$\pi_{X1Y2}(E_1 \bowtie E_2) = E_1 \bowtie \pi_{Y2}(E_2)$$
Se $E_1$ e $E_2$ definite rispettivamente su $X_1$ e $X_2$, $Y_2 \subseteq X_2$ e gli attributi in $X_2 - Y_2$ non sono coinvolti nel join. Non ha effetto sull’efficienza ma sulla leggibilità.

**Distributività della selezione rispetto all'[[Unione]]**
$$\sigma_F(E_1 \cup E_2) = \sigma_F(E_1) \cup \sigma_F(E_2)$$
**Distributività della selezione rispetto alla [[Differenza]]**
$$\sigma_F(E_1 - E_2) = \sigma_F(E_1) - \sigma_F(E_2)$$
**Distributività della proiezione rispetto alla unione**
$$\pi_F(E_1 - E_2) = \pi_F(E_1) - \pi_F(E_2)$$
**Non distributività della proiezione rispetto alla differenza**
In generale:
$$\pi_F(E_1 - E_2) = \pi_F(E_1) - \pi_F(E_2)$$Se $R_1$ e $R:2$ sono definite sull’insieme di attributi $X = AB$, e contengono tuple uguali su A e diverse su B.

**Esempio**
![[Screenshot 2023-12-06 at 14.59.42.png]]
Dipende da chi è A. Se $R_1$ e $R_2$ sono definite su AB e contengono tuple uguali su A e diverse su B, NO.

**Inglobamento di una selezione in un [[Prodotto|prodotto cartesiano]] a formare un join**
$$\sigma_F(R_1 \bowtie R_2) \equiv R_1 \bowtie_F R_2$$
**Altre equivalenze**
![[Screenshot 2023-12-06 at 15.25.52.png]]
Si noti infine che valgono proprietà commutativa e associativa di tutti gli operatori binari (unione, intersezione, join) tranne la differenza.

**Operatori algebrici non insiemistici**
- $\pi^b_{A_i}(R)$: proiezione multiisiemistica (senza eliminazione dei dublicati)
- $\uptau_{\{A_i\}}(R)$: ordinamento

**Calcolo relazionale su ennuple**
Il calcolo relazionale è un linguaggio che permette di definire il risultato di un’interrogazione come l’insieme di quelle ennuple che soddisfano una certa condizione $\phi$.

L’insieme delle matricole degli studenti che hanno superato qualcuno degli esami elencati nella relazione Materie(Materia), si può definire come:
![[Screenshot 2023-12-06 at 15.28.33.png]]

Che è equivalente a:
![[Screenshot 2023-12-06 at 15.28.56.png]]
# References