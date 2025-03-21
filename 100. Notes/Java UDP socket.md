---
Created: 2024-09-26T21:18:00
tags:
  - "#note"
  - padawan
Links: "[[Networks and Laboratory III]]"
Area:
---
# Java UDP socket

[[UDP]] supporta una comunicazione connectionless e fornisce un insieme molto limitato di servizi, rispetto al [[TCP]]. Ha senso usarlo con stream di dati, per esempio audio/video, oppure casi in cui abbiamo degli invii di messaggi con un intervallo regolare.

## Datagram sockets
In java un socket UDP è implementato mediante i `DatagramSocket` 
- non richiede collegamento prima di inviare una pacchetto
- È necessario specificare l'indirizzo [[IP - Internal protocol|IP]] del destinatario per ogni pacchetto spedita
- Ogni pacchetto, chiamato `DatagramPacket` è indipendente dagli altri e porta le info per il suo istradamento

![[Screenshot 2024-09-26 at 21.24.30.png | 500]]

### How to do
1. Aprire il socket, se si sceglie la porta 0 il sistema sceglie una porta libera
```java
DatagramSocket socket = new DatagramSocket(0);
```
2. Impostare un timeout sul socket, opzionale ma consigliato
```java
setToTimeout(15000)
```
- intervallo di tempo che si attende nel receive
- se scade si solleva un eccezione del tipo `InterruptedException`

3. Costruire due pacchetti, uno per inviare richiesta al server ed uno per ricevere una risposta.
```java
InetAddress host = IndetAddress.getByName(HOSTNAME);
DatagramPacket request = new DatagramPacket(new byte[1], 1, host , PORT); 
byte [] data = new byte[1024]; 
DatagramPacket response = new DatagramPacket(data, data.length);
```
4. Mandare la richiesta ed aspettare la risposta
```java
socket.send(request); 
socket.receive(response);
```
- la `socket.send()` non è bloccante
- mentre la `socket.receive()`è bloccante, per evitare attese infinite si mette un **socket timeout**

5. Estrarre i byte dalla risposta e convertirli in String
```java
String daytime = new String(response.getData(),0,response.getLength(),"Us-ASCII"); System.out.println(daytime);
```

### Datagram UDP
- Dimensione massima teorica di un pacchetto 656597 bytes
- In JAVA un datagram UDP è rappresentato come un'instanza della classe DatagramPacket
 
![[Screenshot 2024-09-26 at 21.34.56.png |500]]

### Costruttori
- 2 costruttori per **ricevere dati**
```java
public DatagramPacket(byte[ ] buffer, int length) 
public DatagramPacket(byte[ ] buffer, int offset, int length)
```
- 4 costruttori per **inviare dati**
```java
public DatagramPacket(byte[ ] buffer, int length, InetAddress remoteAddr, int remotePort) public DatagramPacket(byte[ ] buffer, int offset, int length, InetAddress remoteAddr, 
					  int remotePort) 
public DatagramPacket(byte[ ] buffer, int length, SocketAddress destination) 
public DatagramPacket(byte[ ] buffer, int offset, int length, SocketAddress destination)
```


Nel DatagramPacket il buffer viene passato vuoto alla receive che lo riempie con il payload
- **Length** indica il numero i bytes che devono essere copiati dal byte buffer
- **Destination** + **port** individuano il destinatario
- **offset** è un valore che, se settato la copia avviene dalla posizione indicata

```java
DatagramSocket(int p);
```
- costruttore per creazione socket
- crea il socket e lo connette alla porta
- solleva un eccezione se la porta è già occupata
- utilizzato lato client per spedire datagrammi

## Struttura dati
- dati invati mediante UDP rappresentati come **vettori di bytes**
- Si possono usare i **[[Java Stream based IO|filtri]]** per generare streams di bytes a partire da dati strutturati di alto livello
```java
public ByteArrayOutputStream()
public ByteArrayOutputStream(int size)
```

Questi oggetti rappresentato stream di bytes, quando si scrive sullo stream viene riportato in un **buffer di memoria** a **dimensione variabile**. Quando il buffer si riempie la sua dimensione viene raddoppiata.

È possibile collegare ad un `ByteArrayOutputStream` un altro filtro. Questo processo si chiama **serializzazione** e **deserializzazione** 

```java
ByteArrayOutputStream baos= new ByteArrayOutputStream(); 
DataOutputStream dos = new DataOutputStream(baos)
ObjectOutputStream dos = new ObjectOutputStream(baos)
```

![[Screenshot 2024-09-26 at 21.57.01.png | 500]]

I dati presenti nel buffer possono essere copiati in un array di bytes. Flusso dei dati:

![[Screenshot 2024-09-26 at 21.58.52.png | 500]]

## Mutlicasting
L'idea è di inviare i dati ad un host ad un insieme di altri nodi. Solitamente questa operazione avviene a [[Introduction to network layer|livello di network]] dai router, il problema è che non tutti i router supportano i mutlicast. 

A livello di JAVA c'è una Multicast API che mette a disposizione una serie di metodi per simulare questo funzionamento. Si definisce un IP multicast basato sul **concetto di gruppo**, tutti i membri di quel gruppo ricevono il messaggio.
- **unirsi** ad un gruppo mutlicast
- **lasciare** un gruppo
- **spedire** messaggi ad un gruppo
- **ricevere** messaggi indirizzati ad un gruppo

### Caratteristiche
- Utilizza il paradigma connectionles UDP (in caso contrario sarebbero richieste nx(n-1) connessioni per un gruppo di n applicazioni)
- Si perde affidabilità trasmissione
- Implementato nel package `java.net.MulticastSocket`

Creazione di un server multicast:
```java
public class multicast {
	public static void main (String [ ] args) {
		try {
			MulticastSocket ms = new MulticastSocket( );
		} catch (IOException ex) {
			System.out.println("errore"); 
		} 
	}
}
```

Unirsi ad un server multicast:
```java
public class multicast {
	public static void main (String [ ] args) { 
		try {
			MulticastSocket ms = new MulticastSocket(4000); 
			InetSocketAddress ia=InettAddress.getByName("226.226.226.226"); 
			ms.joinGroup (ia); 
		} catch (IOException ex) {
			System.out.println("errore"); 
		}
	}
}
```

Ricevere pacchetti da multicast:
```java
public class provemulticast { 
	public static void main (String args[]) throws Exception { 
		byte[] buf = new byte[10]; 
		InetAddress ia = InetAddress.getByName(“228.5.6.7”); 
		DatagramPacket dp = new DatagramPacket(buf,buf.length); 
		MulticastSocket ms = new MulticastSocket(4000); 
		ms.joinGroup(ia); 
		ms.receive(dp); 
	} 
}
```
- Se si attivano due instanze di provemulticast sullo stesso host **non viene sollevata eccezione**, essa verrebbe sollevata invece se si utilizzasse un `DatagramSocket`
- Esiste la proprietà **reuse socket** che se messa a true da la possibilità di associare più socket alla stessa porta.
```java
try {
	sock.setReuseAddress(true);
} catch (SocketException se) {

}
```
# References