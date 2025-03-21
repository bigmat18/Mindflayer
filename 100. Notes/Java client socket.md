---
Created: 2024-09-26T01:04:00
tags:
  - "#note"
  - youngling
Links: "[[Networks and Laboratory III]]"
Area:
---
# Java client socket

Una [[Introduction to application layer|network application]] corrisponde a due o più processi in esecuzioni su hosts diversi che comunicano e cooperano per realizzare funzionalità globali. Si possono utilizzare due protocolli a livello di [[Introduction to transport layer|trasporto]]
- **connection-oriented**: [[TCP]]
- **connectionless**: [[UDP]]

Per identificare gli host si utilizzano degli [[IP - Internal protocol|IP]] che sono valori formati da 32bit, oppure dei [[DNS|nomi di dominio]] che possono essere associati ad un indirizzo IP.

## Classe InetAddress
```java
public class InetAddress extends Object implements Serializable
```

- Classe in java che gestisce IPv4 e IPv6
- La classe non contiene alcun costruttore
- Utilizza una factory con metodi statici

```java
byte [] getAddress() 
String getHostAddress() 
String getHostName() 
boolean isLoopBackAddress() 
boolean isMulticastAddress() 
boolean isReachable()
```

###### getByName()
Esegue il lookup dell'indirizzo dell'host
```java
InetAddress address = InetAddress.getByName("www.unipi.it");
System.out.println(address);
```
```
Output:
https://www.unipi.it/131.114.21.42
```

###### getLocalHost()
Esegue il lookup dell'indirizzo locale
```java
InetAddress address = InetAddress.getLocalHost(); 
System.out.println(address);
```
```
Output:
DESKTOP-R5C46F3/192.168.1.196
```
### Caching
I metodi visti sopra effettuano il caching fra nome/indirizzo per evitare che ci sia l'accesso al DNS troppo frequentemente, essendo un operazione potenzialmente molto costosa.
- I dati rimangono nella cache 10 secondi se la risoluzione non ha avuto successo
- Se la risoluzione ha avuto successo per un tempo illimitato

## Client socket
Lo standard di comunicazione per connettersi fra servizi è il [[Introduction to application layer|socket]] dove si specifica un IP ed una porta per

![[Screenshot 2024-09-26 at 11.41.59.png | 400]]

In JAVA per la creazione di un socket [[TCP]] locale si utilizza la classe offerta da `java.net.socket`

```java
import java.net.*; 
import java.io.*; 

public class LowPortScanner { 

	public static void main(String[] args) {
		String host = args.length > 0 ? args[0] : "localhost"; 
		
		for (int i = 1; i < 1024; i++) { 
			try { 
				Socket s = new Socket(host, i); 
				System.out.println("There is a server on port " + i + " of " + host);
				s.close(); 
			} catch (UnknownHostException ex) { 
				System.err.println(ex); 
				break; 
			} catch (IOException ex) { 
				// must not be a server on this port 
			}
		}
	}
}
```

In questo esempio il client cerca di creare un socket su ognuna delle prime 1024 porte, in caso fallisca si solleva un eccezione. Per ottimizzare è possibile passare al posto di una stringa, un tipo `InetAddress`-

Una volta stabilita la connessione per scambiarsi i dati si associa uno [[Java Stream based IO|stream]] di input ed uno di output al socket con le seguenti classi.

```java
public InputStream getInputStream () throws IOException 
public OutputStream getOutputStream () throws IOException
```

 In particolare, per poi leggere i caratteri, si utilizzerà questo Stream per create un `InputStreamReader` che traduce caratteri esterni nella codifica interna Unicode

```java
InputStream in = socket.getInputStream(); 
StringBuilder time = new StringBuilder(); 
InputStreamReader reader = new InputStreamReader(in, "ASCII");

for (int c=reader.read();c != -1;) { 
	time.append((char) c); 
}
```
# References