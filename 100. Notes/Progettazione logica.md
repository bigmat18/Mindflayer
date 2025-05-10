**Data time:** 00:45 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Progettazione Basi di Dati]]

**Area**: [[Bachelor's Degree]]
# Progettazione logica

## Trasformazione di schemi
L'obbiettivo della progettazione logica è quello di "tradurre" lo schema concettuale in uno schema logico relazionale che rappresenti gli stessi dati in maniera corretta ed efficiente. Questo richiede una ristrutturazione del modello concettuale.

*Osservazione*: Non si tratta di una pura e semplice traduzione. Infatti alcuni costrutti dello schema concettuale non sono direttamente rappresentabili, nel modello logico è necessario tenere conto delle prestazioni.

I dati in ingresso sono:
- Lo schema concettuale
- Le informazioni sul carico applicativo (dimensioni dei dati e caratteristiche delle operazioni)
- Un modello logico
I dati in uscita invece sono:
- Lo schema logico
- La documentazione associata

La trasformazione di uno schema ad oggetti in uno schema relazionale avviene eseguendo i seguenti passi:
1. Rappresentazione delle associazioni uno ad uno e uno a molti.
2. Rappresentazione delle associazioni molti a molti o non binarie.
3. Rappresentazione delle gerarchie di inclusione.
4. Identificazione delle chiavi primarie.
5. Rappresentazione degli attributi mutlivalore.
6. Appiattimento degli attributi composti.

L'obbiettivo finale è quello di rappresentare le stesse informazioni; minimizzando la ridondanza; e produrre uno schema comprensibile, per facilitare la scrittura e manutenzione delle applicazioni.

**Esempio**. Schema concettuale
![[Screenshot 2023-11-22 at 20.13.14.png]]

Traduzione logica di uno schema
![[Screenshot 2023-11-22 at 20.14.35.png]]

## Rappresentazione delle associazioni
### [[Associazioni 1-N|Uno a molti]]
### [[Associazioni 1-1|Uno ad uno]]
### [[Associazioni N-N|Molti a molti]]


## Traduzione delle gerarchie
il modello relazionale non può rappresentare direttamente le gerarchie. Bisogna eliminare le gerarchie, sostituendole con [[Diagramma delle Classi|classi]] e relazioni:
1. accorpamento delle figlie della gerarchia nel genitore (**relazione unica**).
2. accorpamento del genitore della gerarchia nelle figlie (**partizionamento orizzontale**)
3. sostituzione della gerarchia con relazioni (**partizionamento verticale**).

### [[Relazione Unica in Modello Relazionale|Relazione unica]]
### [[Partizione Orizzontale in Modello Relazionale|Partizionamento orizzontale]]

### [[Partizione Verticale in Modello Relazionale|Partizionamento verticale]]


## Attributi mutlivalore
**Esempio**. Campo multi-valore. Gestione persone con <<più>> indirizzi email.

![[Screenshot 2023-11-22 at 20.52.35.png]]

![[Screenshot 2023-11-22 at 20.53.07.png]]

![[Screenshot 2023-11-22 at 20.53.17.png]]

![[Screenshot 2023-11-22 at 20.53.43.png]]
# References