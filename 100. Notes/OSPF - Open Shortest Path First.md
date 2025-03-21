**Data time:** 21:22 - 19-09-2024

**Status**: #master 

**Tags:** [[ISO-OSI Network layer]]

# OSPF - Open Shortest Path First

- INTRA-AS
- Si utilizza [[Link-state algorithm (LSP)]]
- Metrica = può essere decisa fra latenza, affidabilità, banda, numero di hop ec..

##### OSPF gerarchico
AS può essere partizionato in aree, una delle quali fa da **dorsale**. Questo si fa per ridurre il flooding del link state packet

![[Screenshot 2024-08-26 at 23.14.14.png | 500]]

###### Sistemi interconnessi
Ciascun sistema autonomo sa come inoltrare pacchetti lungo il percorso ottimo verso qualsiasi destinazione interna al gruppo

# References