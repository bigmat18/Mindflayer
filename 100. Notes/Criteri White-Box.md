**Data time:** 14:09 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: [[Bachelor's Degree]]
# Criteri White-Box

Sono criteri per l'individuazione dei casi di input che si basano sulla struttura del codice. Sinonimi: criteri strutturali, criteri a scatola aperta.

Perché criteri basati sul codice?
I criteri strutturali che vediamo oggi devono aiutare ad aggiungere altri test oltre a quelli generati con criteri funzionali. Rispondono alla domanda: “Quali altri casi devo aggiungere per far emergere malfunzionamenti che non sono apparsi con il testing fatto con casi di prova basati su criteri black-box? ”

Per abuso di linguaggio si parla di white/black-box testing: è solo la progettazione white/black box, non il testing!

Banalmente potremmo dire che un programma non è testato adeguatamente se alcuni suoi elementi non vengono mai esercitati dai test. 
I criteri strutturali di progettazione di casi di test (aka control flow testing) sono definiti per classi particolari di elementi e richiedono che i test esercitino tutti quegli elementi del programma. Gli elementi possono essere: comandi, branches (decisioni), condizioni o cammini.

### Grafo di flusso
*Definizione* (**Grafo di flusso**): definisce la struttura del codice identificandone le parti e come sono collegate tra loro, è ottenuto a partire dal codice.

I diagrammi a blocchi (detti anche diagrammi di flusso oflow chart) sono un linguaggio di modellazione grafico per rappresentare algoritmi (in senso lato).

**Esempio**.
```java
double eleva(int x, int y) { 
	if (y<0) 
		pow = 0-y; 
	else 
		pow = y; 
	z = 1.0; 
	while (pow!=0) { 
		z = z*x; 
		pow = pow-1;
	} 
	if (y<0) 
		z = 1.0 / z; 
	return z;
}
```

Grafo di flusso corrispondente al codice:
![[Screenshot 2023-12-04 at 16.07.43.png]]

### [[Criterio con Copertura dei Comandi]]

### [[Criterio con Copertura delle Decisioni]]

### [[Criterio con Copertura dei Cammini]]

### [[Fault based testing]]

# References