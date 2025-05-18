**Data time:** 19:17 - 19-09-2024

**Status**: #master 

**Tags:** [[ISO-OSI Link layer]] [[Networks and Laboratory III]]

**Area**: [[Bachelor's Degree]]
# Interconnection devices

## Repeater
Operano solo a livello fisico, rigenerano il segnale che ricevono ed in passato erano usati per collegare segmenti di Ethernet con topologia a bug

## Hub
Repeater multi-porta (nessuna capacità di filtraggio come repeater)

## Switch
- Operano sia a livello fisico (rigenerando segnale) che a [[Introduction to link layer|livello di link]] (verificando MAC nei frame). 
- Sono dispositivi di [[Store-and-Forward Technique|store e forward]] e non modificano indirizzi MAC in intestazione. 
- Hanno una loro tabella che usano per il filtraggio
- Forniscono trasmissioni multiple simultanee visto che ogni host ha una connessione dedicata, questo viene fatto andando a bufferizzare i pacchetti ed evitando collisioni 

## Ethernet switch
###### Link-layer device con ruolo attivo
Store e forward di frame Ethernet. Esamina gli indirizzi MAC dei frame in arrivo e li inoltra in modo selettivo su uno o più collegamenti
###### Trasparente
Gli host non sono a conoscenza della presenza degli switch
###### Plug-and-plat, self-learning
Gli switch non devono essere configurati

## Router
- Operano a livello **fisico**
- Operano a livello di **[[Introduction to link layer|link]]**
- Operano a livello **[[Introduction to network layer|network]]**

###### Differenze con repeater e switch/hub
1. Router ha 1 indirizzo MAC e 1 indirizzo IP per ogni loro interfaccia
2. Operano solo sui frame in cui indirizzo destinazione è l'indirizzo dell'interfaccia su cui arrivano
3. Cambiano indirizzi link contenuti nei frame che inoltrato

# References