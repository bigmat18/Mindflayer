**Data time:** 00:47 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[DBMS]]

**Area**: [[Bachelor's Degree]]
# Gestione Memoria

### Gerarchia delle memorie
La memoria di un sistema di calcolo è organizzata in una gerarchia. Al livello più alto memorie di piccola dimensione, molto veloci, costose; scendendo lungo la gerarchia la dimensione aumenta, diminuiscono la velocità e il costo.

![[Screenshot 2023-12-08 at 15.40.11.png | 400]]
**Prestazioni di una memoria**: dato un indirizzo di memoria, le prestazioni si misurano in termini di tempo di accesso, determinato dalla somma della:
- [[Parallelization methodology and metrics|latenza]] (tempo necessario per accedere al primo byte)
- e del tempo di trasferimento (tempo necessario per muovere i dati)
$$Tempo \:\: di \:\: accesso = latenza + \large\frac{dimensioni \:\: dati \: da \:\: trasferire}{velocità \:\: di \:\: trasferimento}$$
Un DB, a causa della sua dimensione, risiede normalmente su dischi (ed eventualmente anche su altri tipi di dispositivi). I dati devono essere trasferiti in memoria centrale per essere elaborati dal DBMS.

Il trasferimento non avviene in termini di singole tuple, bensì di blocchi (o pagine, termine comunemente usato quando i dati sono in memoria). Poiché spesso le operazioni di I/O costituiscono il collo di bottiglia del sistema, si rende necessario ottimizzare l’implementazione fisica del DB, attraverso:
- Organizzazione efficiente delle tuple su disco
- Strutture di accesso efficienti
- Gestione efficiente dei buffer in memoria
- Strategie di esecuzione efficienti per le query

### Gli hard disk
Un hard disk (HD) è un dispositivo elettro-meccanico per la conservazione di informazioni sotto forma magnetica, su supporto rotante a forma di piatto su cui agiscono delle testine di lettura/scrittura.

![[Screenshot 2023-12-08 at 15.54.40.png | 500]]

**Il meccanismo del disk drive**
Include organi di registrazione, di posizionamento e di rotazione. Un’unità a dischi contiene una pila di dischi metallici che ruota a velocità costante ed alcune testine di lettura che si muovono radialmente al disco.
![[Screenshot 2023-12-08 at 15.55.37.png]]
Un’unità a dischi contiene una pila di dischi metallici che ruota a velocità costante ed alcune testine di lettura che si muovono radialmente al disco.

Una **traccia** è organizzata in settori di dimensione fissa; i settori sono raggruppati logicamente in **blocchi**, che sono l’unità di trasferimento.

Trasferire un blocco richiede un tempo di posizionamento delle testine, un tempo di latenza rotazionale e tempo per il trasferimento (trascurabile)

**Pagine**
*Definizione*: Un blocco (o **pagina**) è una sequenza contigua di settori su una traccia, e costituisce l’unità di I/O per il trasferimento di dati da/per la memoria principale.

La dimensione tipica di una pagina è di qualche KB (4 -64 KB). Pagine piccole comportano un maggior numero di operazioni di I/O. Pagine grandi tendono ad aumentare la frammentazione interna (pagine parzialmente riempite) e richiedono più spazio in memoria per essere caricate.

Il tempo di trasferimento di una pagina (Tt) da disco a memoria centrale dipende dalla dimensione della pagina (P) e dal transfer rate (Tr).

**Esempio**. Con un transfer rate di 21.56 MB/sec e P = 4 KB si ha Tt= 0.18 ms, con P = 64 KB si ha Tt= 2.9 ms

### Gestore della memoria permanente e del buffer

![[Screenshot 2023-12-08 at 16.01.16.png | 600]]
Il **gestore della memoria permanente** fornisce un’astrazione della memoria permanente in termini di insiemi di file logici di pagine fisiche di registrazioni (blocchi), nascondendo le caratteristiche dei dischi e del sistema operativo.

Il **gestore del buffer** Si preoccupa del trasferimento delle pagine tra la memoria temporanea e la memoria permanente, offrendo agli altri livelli una visione della memoria permanente come un insieme di pagine utilizzabili in memoria temporanea, astraendo da quando esse vengano trasferite dalla memoria permanente al buffer e viceversa.
- Pin count: quando una pagina viene richiesta/rilasciata il contatore viene incrementato/decrementato, 
- dirty/non dirty: la componente che ha eseguito la modifica chiede al buffer di aggiornare la proprietà dirty/nondirty della pagina modificata

![[Screenshot 2023-12-08 at 16.02.48.png | 600]]

**Politiche di rimpiazzamento**
Nei sistemi operativi una comune politica adottata per decidere quale pagina rimpiazzare è la LRU (Least Recently Used), ovvero si rimpiazza la pagina che da più tempo non è in uso.

Nei DBMS, LRU non è sempre una buona scelta, in quanto per alcune query il “pattern di accesso” ai dati è noto, e può quindi essere utilizzato per operare scelte più accurate, in grado di migliorare anche di molto le prestazioni

L’hit ratio, ovvero la frazione di richieste che non provocano una operazione di I/O, indica sinteticamente quanto buona è una politica di rimpiazzamento.

**Esempio**: esistono algoritmi di join che scandiscono N volte le tuple di una relazione. In questo caso la politica migliore sarebbe la MRU (Most Recently Used), ovvero rimpiazzare la pagina usata più di recente.

Questo è un altro motivo per cui i DBMS non usano (tutti) i servizi offerti dai sistemi operativi.

**Struttura di una pagina**
- Struttura fisica: un insieme, di dimensione fissa, di caratteri.
- Struttura logica: informazioni di servizio, un’area che contiene le stringhe che rappresentano i record.
![[Screenshot 2023-12-08 at 16.09.33.png | 400]]
Problema dei riferimenti ai record: coppia (PID della pagina, posizione nella pagina) (RID).

## Gestore strutture di memorizzazione
Ci sono diversi tipi di organizzazione:
- Seriali o Sequenziali
- Per chiave
- Per attributi non chiave

Parametri che caratterizzano un’organizzazione:
- Occupazione di memoria
- Costo delle operazioni di: ricerca per valore o intervallo, modifica , inserzione, cancellazione.

### [[Organizzazione seriale e sequenziale]]

### [[Organizzazione per Chiave]]

### [[Ordinamento di archivi]]

### [[Piani di accesso]]

### [[Transazioni]]
# References