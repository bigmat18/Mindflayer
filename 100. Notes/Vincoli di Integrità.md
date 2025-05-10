**Data time:** 16:01 - 10-05-2025

**Status**: #note #youngling 

**Tags:** [[Basi di Dati]] [[Modello Relazionale]]

**Area**: [[Bachelor's Degree]]
# Vincoli di Integrità

Esistono istanze di basi di dati che, pur sintatticamente corrette, non rappresentano informazioni possibili per l’applicazione di interesse e che quindi generano informazioni senza significato.

**Esempio**
![[Screenshot 2023-11-22 at 18.48.14.png]]

Uno schema relazionale è costituito da un insieme di schemi di relazione e da un insieme di vincoli d’integrità sui possibili valori delle estensioni delle relazioni.

*Definizione*: Un vincolo d’integrità è una proprietà che deve essere soddisfatta dalle istanze che rappresentano informazioni corrette per l’applicazione.

Un vincolo è espresso mediante una funzione booleana (un predicato): associa ad ogni istanza il valore vero o falso.

Essi si usano per: descrizione più accurata della realtà, contributo alla “qualità dei dati”, utili nella progettazione (vedremo), usati dai DBMS nella esecuzione delle interrogazioni, non tutte le proprietà di interesse sono rappresentabili per mezzo di vincoli formulabili in modo esplicito

**Vincoli intra-relazioni**
Sono i vincoli che devono essere rispettati dai valori contenuti nella relazione considerata, vincoli su valori (o di dominio), vincoli di ennupla.

**Vincoli inter-relazionali**
Sono i vincoli che devono essere rispettati da valori contenuti in relazioni diverse.

**Vincoli di ennupla**
I Vincoli di ennupla esprimono condizioni sui valori di ciascuna ennupla, indipendentemente dalle altre ennuple

Un caso particolare è il **vincolo di dominio** che coinvolge un solo attributo:
![[Screenshot 2023-11-22 at 18.55.08.png]]

### [[Chiavi nel Modello Relazionale]]

### [[Vincoli Referenziali]]
# References