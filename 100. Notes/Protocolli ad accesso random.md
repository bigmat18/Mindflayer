####**Data time:** 18:54 - 19-09-2024

**Status**: #master

**Tags:** [[ISO-OSI Link layer]]

**Area**: [[Bachelor's Degree]]
# Protocolli ad accesso random

##  Slotted ALOHA
###### Assunzioni
- Tempo diviso in slot uguali e fissi
- I nodi iniziano a trasmettere all'inizio dello slot (sincronizzati)
- I nodi sono **sincronizzati**
- Se 2 nodi trasmettono insieme, entrambi rilevano la collisione

###### Funzionamento
1. Nodo trasmette frame all'inizio dello slot successivo
2. Verifica se sono avvenute collisione
	- se **NON ci sono collisioni** il nodo può inviare un nuovo frame
	- se **ci sono collisioni** il nodo ritrasmette con probabilità p il frame nello slot successivo finché non ha successo.

![[Screenshot 2024-08-27 at 19.15.27.png | 500]]

###### Pro
- È un protocollo semplice 
- Il nodo attivo può trasmettere alla massima velocità
- È fortemente decentralizzato visto che devono essere sincronizzati solo gli slot.
###### Contro
- Ci sono i problemi delle collisioni, quindi spreco di slot
- Non c'è un gran numero di nodi attivi, il canale usato con successo solo il 37% del tempo.

## ALOHA puro (un-slotted)

Versione più semplice dove **non serve la sincronizzazione**, quando un nodo ha dati da inviare li trasmette subito, questo porta ad una maggior probabilità di collisione. (a tempo t0 collissone a t0-1 e t0+1)

![[Screenshot 2024-09-02 at 18.31.21.png | 400]]

## CSMA - Carrier Sense Multiple Access

- Ascolta prima di inviare
- Se **canale è libero** trasmette l'intero frame
- Se **canale è occupato** ritarda la trasmissione
- Non interrompe qualcuno che invia

#### Funzionamento
1. NIC riceve datagramma dal [[Introduction to network layer|network layer]] 
2. NIC rileva che canale è inattivo avvia trasmissione, se invece è occupato attende
3. Se NIC trasmette tutto messaggio senza interruzioni abbiamo un trasferimento completato
4. Se NIC rileva una trasmissione mentre c'è la sua trasmissione interrompe e invia un segnale di ingorgo
5. Se NIC ha interrotto si attiva il  **backoff binario** dove viene sceso a caso un valore K fra 0 e 2m-1 e si attende K \* 512 prima di tornare al passaggio 2.

#### CSMA/CD - Collision detection
Le collisione possono ancora avvenire a causa del ritardo di propagazione che fa si che due nodi non rilevino le comunicazioni reciproche. In questo caso le trasmissione che collidono **vengono abortite** (carrier sensing)

# References