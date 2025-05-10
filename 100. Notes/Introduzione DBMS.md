**Data time:** 00:46 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[DBMS]]

**Area**: [[Bachelor's Degree]]
# Introduzione DBMS

Un DBMS è un sistema software che **gestisce grandi quantità** di dati persistenti e condivisi. La gestione di grandi quantità di dati richiede particolare attenzione ai problemi di efficienza (ottimizzazione delle richieste, ma non solo!).

La **persistenza** e la **condivisione** richiedono che un DBMS fornisca dei meccanismi per garantire l’affidabilità dei dati (fault tolerance), per il controllo degli accessi e per il controllo della concorrenza.

Diverse altre funzionalità vengono messe a disposizione per motivi di **efficacia**, ovvero per semplificare la descrizione dei dati, lo sviluppo delle applicazioni, l’amministrazione di un DB, ecc.

La **gestione integrata** e la **condivisione dei dati** permettono di evitare ripetizioni (ridondanza dovuta a copie multiple dello stesso dato), e quindi un inutile spreco di risorse (memoria).

Inoltre, la **ridondanza** può dar luogo a problemi di **inconsistenza** delle copie e, in ogni caso, comporta la necessità di propagare le modifiche, con un ulteriore spreco di risorse (CPU e rete).

**Esempio**
il settore Ordini di un’azienda manifatturiera memorizza i propri dati in un file, non condiviso con gli altri settori aziendali. Ogni volta che arriva un ordine, i dati relativi devono essere trasmessi al settore Spedizioni, affinché l’ordine possa essere evaso. A spedizione eseguita, i dati relativi devono essere ritrasmessi al settore Ordini.
![[Screenshot 2023-12-08 at 15.31.33.png]]

Dal punto di vista utente un DB è visto come una collezione di dati che modellano una certa porzione della realtà di interesse.
L’astrazione logica con cui i dati vengono resi disponibili all’utente definisce un **modello dei dati**; più precisamente:

*Definizione* (**modello di dati**): un modello dei dati è una collezione di concetti che vengono utilizzati per descrivere i dati, le loro associazioni/relazioni, e i vincoli che questi devono rispettare.

Un ruolo di primaria importanza nella definizione di un modello dei dati è svolto dai meccanismi che possono essere usati per strutturare i dati (cfr. i costruttori di tipo in un linguaggio di programmazione).
Ad es. esistono modelli in cui i dati sono descritti (solo) sotto forma di alberi (modello **gerarchico**), di grafi (modello **reticolare**), di oggetti complessi (modello a **oggetti**), di relazioni (modello **relazionale**).

## Indipendenza fisica e logica
Tra gli obiettivi di un DBMS vi sono quelli di fornire caratteristiche di:

**Indipendenza fisica**
L’organizzazione fisica dei dati dipende da considerazioni legate all’efficienza delle organizzazioni adottate. La riorganizzazione fisica dei dati non deve comportare effetti collaterali sui programmi applicativi.

**Indipendenza logica**
Permette di accedere ai dati logici indipendentemente dalla loro rappresentazione fisica.

**Architettura semplificata di un DBMS**
![[Screenshot 2023-12-08 at 15.35.45.png | 500]]

![[Screenshot 2023-12-08 at 15.36.28.png | 500]]

# References