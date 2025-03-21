---
Created: 2024-09-26T12:19:00
tags:
  - "#note"
  - padawan
Links: "[[Networks and Laboratory III]]"
Area:
---
# Java NIO

## NIO channel
- comunicazione bidirezionale
- vengono letti e scritti in un buffer
- canali non bloccanti

![[Screenshot 2024-09-26 at 14.09.03.png | 400]]
###### Vantaggi
- definizione primitive più vicine al livello del sistema operativo, aumento performance
- in generale migliori prestazioni
###### Svantaggi
- primitive a più basso livello di astrazione quindi perdità di semplicità

### Scrittura
Se il canale è usato solo in output si può create utilizzando `FileOutputStream`

```java
FileOutputStream fout = new FileOutputStream( "example.txt" ); 
FileChannel fc = fout.getChannel();

// Creazione buffer
ByteBuffer buffer = ByteBuffer.allocate( 1024 );

// Scrittura nel buffer
for (int i=0; i<message.length; ++i)
	buffer.put(message[i]);

buffer.flip();
fc.write(buffer);
```

### Lettura
Se il canale è utilizzato solo in input, possiamo crearlo partendo da un `FileInputStream`

```java
FileInputStream fout = new FileInputStream( "example.txt" ); 
FileChannel fc = fout.getChannel();

// Creazione buffer
ByteBuffer buffer = ByteBuffer.allocate( 1024 );

fc.read(buffer);
```

### Copiare file con NIO

```java
public class ChannelCopy { 
	public static void main (String [] argv) throws IOException { 
		ReadableByteChannel source = Channels.newChannel(newFileInputStream("in.txt")); 
		WritableByteChannel dest = Channels.newChannel (new FileOutputStream("out.txt")); 
		
		hannelCopy1 (source, dest); 
		
		source.close(); 
		dest.close(); 
	}

	private static void channelCopy1 (ReadableByteChannel src, WritableByteChannel dest)
		throws IOException { 
		
		ByteBuffer buffer = ByteBuffer.allocateDirect (16 * 1024); 
		while (src.read (buffer) != -1) { 
			// prepararsi a leggere i byte che sono stati inseriti nel buffer 
			buffer.flip(); 
			// scrittura nel file destinazione; può essere bloccante 
			dest.write(buffer); 
			// non è detto che tutti i byte siano trasferiti, dipende da quanti 
			// bytes la write ha scaricato sul file di output 
			// compatta i bytes rimanenti all'inizio del buffer 
			// se il buffer è stato completamente scaricato, si comporta come clear
			buffer.compact(); 
		} 
		// quando si raggiunge l'EOF, è possibile che alcuni byte debbano essere ancor 
		// scritti nel file di output 
		buffer.flip(); 
		
		while (buffer.hasRemaining()) { 
			dest.write (buffer); 
		}
	}
}
```
## ByteBuffer
- Implementati nella classe `java.nio.buffer`
- Contengono dati appena letto o che devono essere scritti su un channel
- Composti da uno spazio di memorizzazione: byte buffer
- Ed un insieme di variabili di stato

![[Screenshot 2024-09-26 at 13.43.27.png | 500]]

###### Capacity
Massima capacità di elementi del buffer, definita al momento della creazione del buffer e non può essere modificata, se si tenta a leggere o scrivere in una posizione maggiore si solleva un eccezione
###### Limit
Indica il limite ella porzione del buffer che può essere letta/scritta, di default limit=capacity per la scrittura, mentre la la lettura delimita la porzione di buffer che contiene dati effettivi. Viene aggiornata da sola dopo un inserimento
###### Position
È come un file pointer per un file ad accesso sequenziale, posizione in cui bisogna scrivere o da cui bisogna leggere. Viene aggiornato implicitamente durante le operazioni di lettura e scrittura
###### Mark
Memorizza il puntatore alla posizione corrente. Vale sempre che
$$0 \leq mark \leq position \leq limit \leq capacity $$

#### Scrittura dati nel buffer

![[Screenshot 2024-09-26 at 13.51.25.png |400]]

#### Flipping del buffer

![[Screenshot 2024-09-26 at 13.53.07.png |400]]
#### Lettura dal buffer

![[Screenshot 2024-09-26 at 13.53.23.png|400]]
#### Operazione Mark

![[Screenshot 2024-09-26 at 13.53.50.png | 400]]
#### Operazione reset

![[Screenshot 2024-09-26 at 13.54.10.png | 400]]
#### Operazione clearing

![[Screenshot 2024-09-26 at 13.54.25.png |400]]
#### Operazione rewinding

![[Screenshot 2024-09-26 at 13.54.51.png | 400]]
#### Operazione compacting

![[Screenshot 2024-09-26 at 13.55.11.png | 400]]

#### Non direct buffers
![[Screenshot 2024-09-26 at 13.57.49.png | 400]]

```java
ByteBuffer buf = ByteBuffer.allocate(10);
```

Quando si crea un buffer viene creato un oggetto nel heap, abbiamo una doppia copia dei dati, nel kernel e nel heap della JVM. Questa processo è molto costoso e può essere ottimizzato.

#### Direct buffer

![[Screenshot 2024-09-26 at 14.00.19.png | 400]]

```java
ByteBuffer buf = ByteBuffer.allocateDirect(10);
```

Trasferisce i dati tra il programma ed il sistema operativo mediante accesso diretto alla kernel memory da parte della JVM, evitando copie di dati da/in buffer itermedi, questo migliora le performance

## Socket channel
È un channel associato ad un socket [[TCP]] che "cambia" un socket con un canale di comunicazione bidirezionale.
- scrive e legge su socket TCP
- estende `AbstractSelectableChannel` ed è grazie a questo **non bloccante**
- se si mette in modalità bloccate ha un funzionamento simile a quello degli stream socket

```java
ServerSocketChannel serverSocketChannel = ServerSocketChannel.open(); 
ServerSocket socket = serverSocketChannel.socket(); 
socket.bind(new InetSocketAddress(9999)); 
serverSocketChannel.configureBlocking(false); 
while(true){ 
	SocketChannel socketChannel = serverSocketChannel.accept(); 
	if(socketChannel != null)
		//do something with socketChannel... 
	else 
		//do something useful... 
}
```

- Si associa un oggetto di tipo Socket
- La creazione può essere
	- **implicita** se si accetta una connessione su un `ServerSocketChannel`
	- **esplicita** (lato client) quando si apre una connessione verso un server
- Il non blocking lato client significa ad esempio nel caso di un app con GUI che si continua a gestire l'interazione con l'utente

`finishConnect()` per controllare la terminazione dell'operazione

```java
socketChannel.configureBlocking(false); 
socketChannel.connect(new InetSocketAddress("www.google.it", 80)); 
while(! socketChannel.finishConnect() ){ 
	//wait, or do something else... 
}
```

## Multiplexed I/O
Quando si ha un server che gestisce un grande numero di connessioni con un thread per ogni connessioni può succedere che ci siano problemi di scalabilità visto che il tempo di context switching può essere la maggior parte del tempo impiegato.

![[Screenshot 2024-09-26 at 12.53.10.png |500]]
La soluzione è usare **non blocking I/O** con notifiche bloccanti.
1. L'app registra "descrittori" delle operazioni I/O
2. L'app esegue una operazione di **monitoring di canali**
	- System call bloccante
	- Restituisce descrittore quando indica che un operazione I/O è pronta
	- A quel punto si effettua una read non bloccante

Abbiamo quindi un unico thread che gestisce un numero arbitrario di sockets, questo migliora le performance e la scalabilità, ma è più complesso da gestire

![[Screenshot 2024-09-26 at 12.57.59.png | 600]]

### Selector
È un componente che esamina uno o più NIO channels e determina quali canali sono pronti per leggere/scrivere. Esso selezione un SelectableChannel pronto per operazioni di rete

```java
...
Selector selector = Selector.open();
channel.configureBlocking(false); 
Selectionkey key = channel.register(selector, ops, attach);
```
- **ops** indica quali eventi si è interessati a monitorare su quel canale. Per esempio `SelectionKey.OP_READ`
- **attach** è un buffer associato ad un canale.
- **SelectionKey** è un token che rappresenta un oggetto di tipo SelectionKey che valida fino a che non viene cancellata eseplicitamente

### SelectionKey
È il risultato della registrazione di un canale su un selettore e memorizza
- Il canale a sui si riferisce
- Il selettore a cui si riferisce
- **l'interest set** utilizzato quando viene invocato il metodo select per monitorare i canali del selettore, definisce le operazioni da dover are
- **il ready set** contiene gli eventi che sono pronti su quel canale
- un allegato **attachment** spazio di memorizzazione associato a quel canale per quel selettore

###### Ready set
Aggiornato quando si esegue una operazione di monitoring sui canali mediante una select ed identifica le 
chiavi per cui il canale è pronto per l'esecuzione

![[Screenshot 2024-09-26 at 13.29.20.png | 500]]

###### Attachment
Utile quando si vuole accedere ad informazioni relative al canale che riguarda il suo stato pregresso, consente di tener traccia di quanto è stato fatto in una operazione precedente ed è necessario perché le operazioni di lettura e scrittura non bloccanti non possono essere considerate atomiche
### Insiemi di chiavi
Ogni oggetto selettore mantiene al suo interno i seguenti insiemi di chiavi.
###### Key set
Insieme delle SelectionKey associate al selettore
###### Selected Key set
Insieme di chiavi precedentemente registrate tali per cui una delle operazioni è pronta per l'esecuzione
###### Cancelled Key Set
Chiavi che sono state cancellate con il metodo `cancel()` ma il cui canale è ancora registrato sul selettore
### Select / SelectNow

```java
// bloccante
int select.select();
// bloccante con timeout
int select.select(long timeout);
// non bloccante
int select.selectNow();
```

Metodo  bloccante finché almeno un canale non è pronto, selezione tra i canali registrali quelli pronti per almeno un operazione e restituisce il numero di canali pronti.

1. Cancella ogni chiave appartiene al Cancelled Key Set
2. Interagisce con il sistema operativo per verificare lo stato di readliness di ogni canale registrato, per ogni operazione
3. Per ogni canale con almeno un operazione ready 
	- Se il canale già esiste nel **SelectedKeySet** aggiorna il **ready set** della chiave corrispondente calcolando l'or bit a bit tra il valore precedente e la nuova maschera
	- Altrimenti resetta il ready set e lo imposta con la chiave dell'operazione pronta

```java
Selector selector = Selector.open();
channel.configureBlocking(false); 
SelectionKey key = channel.register(selector, SelectionKey.OP_READ); 

while(true) { 
	// variante non bloccante
	int readyChannels = selector.selectNow(); 
	if(readyChannels == 0) 
		continue; 
	
	Set selectedKeys = selector.selectedKeys(); 
	Iterator keyIterator = selectedKeys.iterator(); 
	
	while(keyIterator.hasNext()) { 
		SelectionKey key = keyIterator.next(); 
		
		if(key.isAcceptable()) { 
			// a connection was accepted by a ServerSocketChannel. 
		} else if (key.isConnectable()) { 
			// a connection was established with a remote Server (client side) 
		} else if (key.isReadable()) { 
			// a channel is ready for reading 
		} else if (key.isWritable()) { 
			// a channel is ready for writing 
		} 
		
		keyIterator.remove(); 
	} 
}
```
# References