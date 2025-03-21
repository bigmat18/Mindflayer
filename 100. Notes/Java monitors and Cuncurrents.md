---
Created: 2024-09-25T23:45:00
tags:
  - "#note"
  - padawan
Links: "[[Networks and Laboratory III]]"
Area:
---
# Java monitors and Cuncurrents

Quando abbiamo che un insieme di thread vuole accedere ad una stessa risorsa si possono creare errori ed inconsistenze, per evitarlo possiamo utilizzare delle **lock implicite** o utilizzo di **synchronized** 

```java
public synchronized void someMethod() {// Do work}
```

- Quando viene invocato il metodo si tenta di acquisire un lock associato all'oggetto, e quando il metodo finisce il lock viene rilasciato

I **monitors** sono una classe di oggetti utilizzabili in modo thread save e si implementa grazie ad una lock implicita in ogni oggetto java. La coda di attesa è gestita dalla JVM.

![[Screenshot 2024-09-25 at 23.53.19.png |500]]

- i costruttori non devono essere dichiarati `synchronized`, in caso contrario si solleva un eccezione
- `synchronized` non è ereditato da overriding
- e non ha senso specificare `synchronized` nelle interfacce essendo legato ad un oggetto

###### void wait()
Sospende il thread fino a che un altro thread non invoca notify() o notifyAll(). Implementa un attesa passiva e rilascia il lock quando viene chiamato.
###### void notify()
Sveglia un singolo thread in attesa su un determinato oggetto, nessuno se non ci sono thread in attesa.
###### void notifyAll()
Sveglia tutti i thread in attesa su questo oggetto, i vari thread andranno poi a competere per andare ad acquisire la lock.

###### Vantaggi monitor
- L'unità di sincronizzazione è il metodo in questo modo tutte le sincronizzazioni sono facilmente visibili guardando la signature del metodo
- È un costrutto strutturato, diminuisce quindi la complessità del programma concorrente

###### Svantaggi monitor
- "coarse grain" synchronization che diminuisce il livello di concorrenza

## Lock intrinsechi
Se non si intende sincronizzare un intero metodo si può sincronizzare solo un blocco di codice all'interno di un metodo. Risolve alcuni problemi del synchronized sul metodo.

```java
public void foo() { 
	synchronized(this){ 
	} 
}
```

- stesse proprietà del synchronized applicato al metodo ma vengono attivate solo per quel blocco
- si usa this per utilizzare il lock dell'oggetto
- Possibilità di utilizzare altri oggetti

```java
public void foo() { 
	synchronized(obj){ 
		condition = ...;
		obj.notifyAll();
	} 
}
```

## Java collections
Insieme di classi che consentono di lavorar con gruppi di oggetti, ovvero **collezioni di oggetti**. Per esempio abbiamo le interfacce:
- **set**: una collezione dove ciascun valore appare una sola volta
- **list**: collezione ordinata di valori, anche duplicati
- **map**: collezione dove c'è il mapping da chiavi a valori e le chiavi sono uniche

### Java iterators
Usati per accedere agli elementi di una collezione una alla volta. L'interfaccia `Collection` contiene il metodo `iterator()` che restituisce un iteratore per una collezione.

```java
public class PersonList { 
	public static void main (String args[]) { 
		Person Tom = new Person("Tom", 45, "professor"); 
		Person Harry = new Person("Harry", 20, "student"); 
		List pList=new LinkedList ();
		...
		Iterator tIterator = pList.iterator(); 
		while (tIterator.hasNext()) { 
			Person tPerson = (Person) tIterator.next(); 
			System.out.println(tPerson); 
		}
	}
}
```

### Collections and thread safe
Non tutte le collezioni sono thread safe, queste non offrono alcun supporto per la sincronizzazione. Per esempio collezioni non thread safe sono `ArrayList`, mentre le thread safe sono `Vector, Hashtable`

Per risolvere il problema di concorrenza è possibile
- Andare ad incapsulare ogni in un blocco syncronized
- Trasformare la collection in una thread-safe
```java
addElements(Collections.synchronizedList(new ArrayList()));
```
Questo metodo trasforma al collection in thread-safe
- Usare un'unica mutual exclusion lock intrinseca per tutta la collezione gestita dalla JVM

## Cuncurrent collections
- Sono un evoluzione delle precedenti librerie basata sulla esperienza nel loro utilizzo ed utilizzano delle lock a fine-grain (non bloccano l'intera collezione)
- Forniscono inoltre alcune utili operazioni atomiche composta da operazioni elementari
- iteratori **fail safe/weakly consistent** restituiscono tutti gli elementi che c'erano quando l'iteratore è stato creato mentre restituisce o meno elementi aggiunti in concorrenza

```java
import java.util.concurrent.ConcurrentHashMap;

ConcurrentHashMap map = new ConcurrentHashMap<String, Integer>();
```

- tipo di hashmap thread-safe che utilizza strategie di lock migliori
- permette modifiche simultanee se si modificano segmenti diversi
# References