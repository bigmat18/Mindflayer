**Data time:** 19:14 - 19-09-2024

**Status**: #master 

**Tags:** [[ISO-OSI Link layer]]

**Area**: [[Bachelor's Degree]]
# Ethernet

## Caratteristiche 
Posizione dominante nel mercato della comunicazione di LAN cablate. È stata la prima LAN ad alta velocità a vasta diffusione.
##### Non affidabile (unreliable)
Nodo in ricezione non invia ack o nack al nodo mittente quindi i dati nel frame eliminati sono recuperati solo se il trasferimento dati affidabile è implementato ai livelli superiori. Si usa [[Protocolli ad accesso random |CSMA/CD]] con binary backoff
##### Connectionless
No handshaking tra nodo mittente e destinatario.

## Struttura pacchetti Ethernet

![[Screenshot 2024-08-27 at 22.48.34.png | 500]]
###### Preambolo (8 byte)
Sono 8 byte che servono per "attivare" gli adattatori dei riceventi e sincronizzare i loro orologi con quello del trasmittente.
###### Dati (46 byte)
L'unità massima di trasferimento MTU varia da 46 byte ad un massimo di 1500 byte. Se il datagramma è più grande va frammentato
###### Indirizzo di destinazione (6 byte)
Quando un adattatore riceve un pacchetto con un indirizzo di destinazione trasferisce il contenuto del campo dati del pacchetto al livello di rete
###### Indirizzo sorgente (6 byte)
Indirizzo dell'adattatore (scheda di rete) che trasmette il frame
###### Campo di tipo (2 byte)
Consente a Ethernet di supportare vari protocolli di rete (IP, ARP etc..)
###### Controllo CRC
Consente all'adattatore di rilevare la presenza di un errore nei bit del pacchetto

## Ethernet standards
###### Ethernet
Primo standard, velocità di 10 Mbps.
###### Fast Ethernet
100 Mbps, 100 metri di distanza su doppino e a parecchi chilometri su fibra.
###### Gigabit Ethernet
1000 Mbps = 1 Gbps. Mantiene invariata lunghezza min/max frame
###### 40/100 Gigabit Ethernet

# References