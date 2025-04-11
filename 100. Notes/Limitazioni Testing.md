**Data time:** 13:54 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: 
# Limitazioni Testing

Il testing è una tecnica di verifica ed è come le altre sottoposta al problema dell'indecibilità.

Una **prova formale di correttezza** corrisponderebbe all’esecuzione del sistema con tutti i possibili input.

**Testing esaustivo**
Fare testing esaustivo vuol dire eseguire e provare ogni possibile input del programma. Esso richiederebbe però:
- un tempo infinito, se gli input sono infiniti (oltre ad esserci in questi casi limiti fisici di memoria).
- un tempo troppo lungo, per domini di input finiti ma molto grandi per un programma che fa la somma di due int ci vorrebbero:
$$2^{32} \times 2^{32} = 2^{64} \approx 10^{21}$$
Test. Ipotizzando 1 nanosecondo per ogni esecuzione:$$10^{21} \times 10^{-9} = 10^{12} \approx 30.000 \:\: anni$$
*Definizione* (**Tesi di Dijkstra**): il test di un programma può rilevare la presenza di difetti, ma non dimostrarne l'assenza.
# References