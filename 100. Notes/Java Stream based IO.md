---
Created: 2024-09-25T23:58:00
tags:
  - "#note"
  - padawan
Links: "[[Networks and Laboratory III]]"
Area:
---
# Java [[Channels in Message Passing|Stream based IO]]

## Caratteristiche generali java stream

![[Screenshot 2024-09-26 at 00.02.25.png |400]]

- va a definire un insieme di **astrazioni per la gestione dell'I/O** 
- Si applicato su diversi tipi di device di input/output
- L'accesso agli stream è sequenziale
- Mantengono un ordinamento di tipo FIFO
- sono **one-way** quindi o read-only o write-only
- Sono **bloccanti**
- Non è richiesta corrispondenza fra lettura e scrittura

Il package `java.io` ha l'obbiettivo di fornire un'astrazione che incapsuli tutti i dettagli del dispositivo sorgente/destinatario, fornendo un modo semplice per aggiungere ulteriori funzionalità di quelle basi creando un **approccio a livelli**

![[Screenshot 2024-09-26 at 00.09.26.png |400]]
## Java filter
Le classi `InputStream` e `OutputStream` operano su row bytes e vanno ad effettuare crittografia, compressione, buffering, traduzione dei dati in un formato

I filtri invece (**Readers/Writers**) sono orientati al testo e permettono di decodificare bytes in caratteri. Essi sono organizzati in catene.

![[Screenshot 2024-09-26 at 00.11.36.png | 500]]

Esempio di codice java in cui si usa una catena di filter per andare a leggere dei dati da un file.
```java
import java.io.*; 
public class TestDataIOStream { 
	public static void main(String[] args) { 
		String filename = "data-out.dat"; 
		// Write primitives to an output file 
		try (DataInputStream in = 
					new DataInputStream( 
						new BufferedInputStream( 
							new FileInputStream(filename)))) { 
			System.out.println("byte: " + in.readByte()); 
			System.out.println("short: " + in.readShort()); ...
		}
	}
}
```
# References