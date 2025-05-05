**Data time:** 21:21 - 19-09-2024

**Status**: #master 

**Tags:** [[ISO-OSI Network layer]]

**Area**: [[Bachelor's Degree]]
# RIP - Routing Information Protocol

- INTRA-AS
- Metrica = numero sotto reti attraversate
- Si usa [[Distance vector algorithm]] con Poisoned Reverse (inf = 16)
- Usa [[UDP]]
###### Algoritmo
1. Un nodo invia la tabella di inoltro ai propri vicini
2. Un nodo R riceve annuncio da vicino V con $D_V[Y]$ e fa $D_R[Y] = 1 + D_V[Y]$
3. Il nuovo percorso viene inserito in tabella
	- se è un percorso non presente
	- se $D_V[Y] + 1 < D_{Rold}[Y]$ (costo ricevuto inferiore a quello vecchio percorso)
	- se V è nextHop del vecchio e nuovo percorso ma il corso è cambiato
# References