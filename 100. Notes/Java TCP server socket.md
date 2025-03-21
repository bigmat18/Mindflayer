---
Created: 2024-09-26T11:50:00
tags:
  - "#note"
  - youngling
Links: "[[Networks and Laboratory III]]"
Area:
---
# Java TCP server socket

## Descrizione generale
Esistono due tipi di socket [[TCP]] lato server
- **welcome sockets** (passive, listening): il socket server accetta le richieste di connessione
- **connection sockets** (active): il server si connette ad un particolare client

Le operazioni che vengono svolte sono le seguenti
1. Client crea un active socket
2. Il server accetta una richiesta di connessione su **welcome socket** creando un proprio socket che rappresenta il punto terminale della connessione con il client
3. La comunicazione avviene con la **copia di active socket** presenti nel client e nel server

![[Screenshot 2024-09-26 at 12.07.49.png | 400]]
4. Dopo la richiesta di connessione vengono associati degli [[Java Stream based IO|streams]] di input e output unidirezionali
5. Avviene la comunicazione medianti letture e scritture di dati sullo stream.

![[Screenshot 2024-09-26 at 12.11.31.png | 500]]

## Java server
In java c'è il package `java.net.ServerSocket` che fornisce dei costruttori per i socket.

```java
public ServerSocket(int port)throws BindException, IOException 
public ServerSocket(int port,int length) throws BindException, IOException
public ServerSocket(int port,int length,Inetaddress bindAddress)
```

- **port** indica la porta in cui si deve mettere in ascolto
- **length** indica la lunghezza della coda in cui vengono memorizzare le richieste di connessione
- **bindAddress** permette di collegare il socket ad uno specifico indirizzo IP locale

Successivamente si utilizzate il metodo `accept` per accettare una nuova connessione dal **connection socket**. Questo metodo mette il server in attesa di una connessione, è un metodo bloccante
```java
public Socket accept( ) throws IOException
```

```java
// instantiate the ServerSocket 
ServerSocket servSock = new ServerSocket(port); 
while (! done) { 
// accept the incoming connection 
Socket sock = servSock.accept(); 
// ServerSocket is connected ... talk via sock 
InputStream in = sock.getInputStream(); 
OutputStream out = sock.getOutputStream(); 

//client and server communicate via in and out and do their work 
sock.close(); 
} 
servSock.close();
```
# References