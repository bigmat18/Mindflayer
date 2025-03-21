**Data time:** 01:10 - 20-09-2024

**Status**: #master 

**Tags:** [[Networks and Laboratory III]]

# Peer-to-Peer
Paradigma in cui tutti gli host sono peer e agiscono sia da client che da server
- Tutti i nodi hanno la stessa importanza
- Nessun controllo centralizzato
- **Servent** = server + client (anche se possono in realtà essere presenti server centralizzati o nodi con funzionalità diverse)
- Sistemi **altamente distribuiti**
- Nodi molto **dinamici** e **autonomi**
- Operazioni di ingresso/uscita della rete anche sofisticate

## Reti centralizzate
Modello **Directory centralizzata** dove abbiamo appunto una directory che fornisce la lista degli [[IP - Internal protocol|IP]] dei vari nodi per inizializzare la comunicazione.
- Abbiamo un unico punto di fallimento
- E collo di bottiglia

## Reti decentralizzate
Non abbiamo nessun servizio centrale ma i peer si organizzano i una **overlay network**

### Reti non strutturate
- Nodi organizzati come grafo in modo random
- Aggiunta e rimozione nodi molto facile
- Non ci sono vincoli sul posizionamento delle risorse rispetto alla topologia del grafo
#### Query flooding
- Peer invia query a tutti i suoi vicini
- Se peer riceve query e ha il file richiesto invia messaggio **QueryHit**, altrimenti lo inoltra

![[Screenshot 2024-09-20 at 01.18.19.png | 300]]

#### Copertura gerarchica
- Si cerca di combinare il meglio delle reti centralizzate e le reti decentralizzate non strutturate
- Un peer è o un **group leader** o viene **assegnato ad un group leader**
- Si usa una connessione [[TCP]] tra peer ed il suo group leader e fra i vari group leader

Quando arriva richiesta (nel quale c'è l'hash del file richiesto) a group leader, esso conosce cosa contegono i nodi sotto di lui, e se uno contiene i dati risposte con \<hash del fle, indirizzo IP>

### Reti strutturate
- Sistemi con **Distribured Hash Table** 
- Ad ogni peer è associato ID ed ogni peer conosce un certo numero di peer
- Ad ogni risorsa condivisa viene assegnata un ID basato su funzione hash applicata al contenuto
- Routing avviene verso il peer cha ha l'ID più "vicino" a quello della risorsa

###### Conseguenze
- Vincoli sul grafo e sul posizionamento
- Organizzazione rete con principi più rigidi
- Aggiunta e rimozione nodi più difficile
- Migliora la localizzazione delle risorse

## Bittorrent
Protocollo molto diffuso per la distribuzione di file in Internet che si basa su dividere un file in pezzi (**chunk**) e far distribuire ad ogni peer i dati ricevuti, fornendoli a nuovi destinatari.
- Riduce carico ad ogni sorgente
- Riduce la dipendenza dal distributore originale
- Si fornisce ridondanza

###### Tracker
Il nodo che coordina la distribuzione del file. Tiene traccia dei peer che stanno partecipando al torrent.
###### File .torrent
Per condividere un file un peer cerca un file .torrent, che contiene metadati sul file condiviso e sul tracker (il file contiene, info e announce)
###### Nuovo peer
Quando c'è un nuovo peer, cerca e ottiene un file .torrent, ovvero un metafile con info sui chunk e indirizzo IP di tracker. Contatta tracker e riceve indirizzi di alcuni peer.

- Un peer quando entra a far parte di un torrent per la prima volta non ha un chunk file
- Con il tempo accumula più parti che invia ad altri peer
- Quando ha l'intero file può lasciare il torrent o rimanere nel torrent per continuare ad inviare chunck
- Si può lasciare il torrent in qualsiasi momento
# References