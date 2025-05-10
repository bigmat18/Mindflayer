**Data time:** 00:45 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Normalizzazione]]

**Area**: [[Bachelor's Degree]]
# Introduzione Normalizzazione

Ci sono due metodi per produrre uno schema relazione:
1. Partire da un buon schema a oggetti e tradurlo
2. Partire da uno schema relazionale fatto da altri e modificarlo o completarlo.

Teoria della progettazione relazionale: studia cosa sono le “anomalie” e come eliminarle **(normalizzazione)**. 
È particolarmente utile se si usa il metodo (1). È moderatamente utile anche quando si usa il metodo (2).

### Schemi con anomalie
**Esempio**
StudentiEdEsami(Matricola, Nome, Provincia,AnnoNascita, Materia, Voto)

**Le anomalie sono:** 
Ridondanze, Potenziali inconsistenze, Anomalie nelle inserzioni, Anomalie nelle eliminazioni.

**Soluzione:** dividiamo lo schema in due tabelle. 
Studenti (Matricola, Nome, Provincia, AnnoNascita), Esami (Nome, Materia, Voto)
### Obbiettivi
Nozione base: dipendenze funzionali. Obiettivi della teoria:
- **Equivalenza** di schemi: in che misura si può dire che uno schema rappresenta un altro.
- **Qualità** degli schemi (forme normali).
- **Trasformazione** degli schemi (normalizzazione di schemi)

Ipotesi dello schema di relazione universale: Tutti i fatti sono descritti da attributi di un’unica relazione (relazione universale), cioè gli attributi hanno un significato globale.

*Definizione*: Lo schema di **relazione universale** U di una base di dati relazionale ha come attributi l’unione degli attributi di tutte le relazioni della base di dati.

*Definizione*: Una **forma normale** è una proprietà di una base di dati relazionale che ne garantisce la “qualità”, cioè l'assenza di determinati difetti.

Quando una relazione non è normalizzata: 
- presenta ridondanze, 
- si presta a comportamenti poco desiderabili durante gli aggiornamenti

La normalizzazione è una procedura che permette di trasformare schemi non normalizzati in schemi che soddisfano una forma normale.

**Perché questi fenomeni sono indesiderabili?**
![[Screenshot 2023-12-06 at 15.47.56.png]]
- **Ridondanza**: Lo stipendio di ciascun impiegato è ripetuto in tutte le ennuple relative. Questo perché lo stipendio dipende solo dall’Impiegato. Il costo del bilancio per ogni progetto è ripetuto.
- **Anomalia di aggiornamento**: Se lo stipendio di un impiegato varia, è necessario andarne a modificare il valore in diverse ennuple.
- **Anomalia di cancellazione**: Se un impiegato interrompe la partecipazione a tutti i progetti, dobbiamo cancellare tutte le ennuple in cui appare, e in questo modo l’impiegato non sarà più presente nel database.
- **Anomlia di inserimento**: Un nuovo impiegato non può essere inserito finché non gli viene assegnato un progetto.

### Linee guida per una corretta progettazione
**Semantica degli attributi**
Si progetti ogni schema relazionale in modo tale che sia semplice spiegarne il significato. Non si uniscano attributi provenienti da più tipi di classi e tipi di associazione in una unica relazione.

**Ridondanza**
Si progettino gli schemi relazionale in modo che nelle relazioni non siano presenti anomalie di inserimento, cancellazione o modifica. Se sono presenti delle anomalie (che si vuole mantenere), le si rilevi chiaramente e ci si assicuri che i programmi che aggiornano la base di dati operino correttamente.

**Valori nulli**
Per quanto possibile, si eviti di porre in relazione di base attributi i cui valori possono essere (frequentemente) nulli. Se è inevitabile, ci si assicuri che essi si presentino solo in casi eccezionali e che non riguardino una maggioranza di tuple nella relazione.

**Tuple spurie**
Si progettino schemi di relazione in modo tale che essi possano essere riuniti tramite JOIN con condizioni di uguaglianza su attributi che sono o chiavi primarie o chiavi esterne in modo da garantire che non vengano generate tuple spurie. Non si abbiano relazioni che contengono attributi di «accoppiamento» diversi dalle combinazioni chiave esterna-chiave primaria.
# References