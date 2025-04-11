**Data time:** 13:51 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: 
# Introduzione Verifica Validazione

Nel 1937 Alan turing ha dimostrato che alcuni problemi non possono essere risolti da un algoritmo (programma). Tali problemi sono quelli che coinvolgono il problema della terminazione.

*Definizione* (**Problema terminazione**): esiste un algoritmo/programma che presi in ingresso un qualsiasi altro programma e un input, stabilisce se il programma su tale input termina o no?

Si assuma che esista un programma C che prende in input un programma (a) ed un input per a che chiamiamo b.
```c
bool C(a,b) {return halts(a(d));}
```
dato che un programma è a sua volta una sequenza di caratteri, si può invocare C(a,a).
Si può quindi definire K(a) come segue:
```c
bool K(a) {
	if (C(a,a)) while(true) skip;
	else return false;
}
```
A questo punto possiamo dire che:
Il programma K con input K termina (restituendo false), se e solo se C(K,K) è falso, ma C(K,K) è falso solo se halts(K(K)) è falso, va a dire se il programma K con input K non termina. K(K) termina se e solo se K(K) non termina => paradosso => Non può esistere il programma C.

In particolare,dobbiamo concludere che non esiste un programma P che per ogni programma Q e input D, dice se il programma Q sull’input D termina o no (**Halting Problem**).
```c
while (x>0) 
	x = x + 1;
y = 27;
```
Purtroppo non è solo un risultato teorico: quasi tutte le proprietà interessanti dei programmi incorporano l’halting problem.

**Indecidibilità**
Esistono programmi che è possibile dimostrare corretti in tempo finito.
```java
public void printHW() {
	System.out.println("Hello, World");
}
```
Ne esistono altri per cui non è possibile. Quindi non esiste alcun programma P che prende in input altri programmi e per ognuno di questi decide in tempo finto se è corretto o meno.

**Esempio**. È indecidibile dire con un algoritmo generale se un generico programma C vada in ciclo infinito su un generico input.
```java
public void printC(myHome Home){
	Calzini calzini = myhome.calzini;
	while (!appaiati(calzini))
		calzini = appaia(calzini);
	System.out.println("Hello, world");
}
```

Per risolvere questo problema possiamo usare per esempio le **Triple di Hoare**.
Dove si nasconde il problema? La logica al primo ordine è indecidibile. In altre parole: Esiste un algoritmo che, per ogni formula $F$ in logica al primo ordine, mi permetta di decidere in tempo finito se $F$ è valida o meno? se cioè $\vDash F$ oppure $\nvDash F$? 

No, non esiste. Si possono enumerare (scrivere una dopo l’altra) tutte le formule valide, ma in tempo finito posso non arrivare a scrivere né $F$ né $\lnot F$.

**Concetti e terminologia** 
Visualizzare il "quadro generale" della qualità del software nel contesto di un progetto di sviluppo software e organizzazione: 
- attività di verifica e di validazione (V&V: verification and validation) del software. 
- La selezione e la combinazione di attività di V&V all'interno di un processo di sviluppo software.

Il sw ha alcune caratteristiche che rendono V&V particolarmente difficile.
- requisiti di qualità diversi 
- il sw è sempre in evoluzione 
- distribuzione irregolare dei guasti 
- non linearità, esempio: 
	- Se un ascensore può trasportare un carico di 1000 kg, può anchetrasportare qualsiasi carico minore. 
	- Se una procedura ordina correttamente un set di 256 elementi, potrebbe non riuscire su un set di 255 o 53 o 12 elementi, non ché su 257 o 1023.

**Dipendenze dai linguaggi**
Nuovi approcci di sviluppo possono introdurre nuovi tipi di errori. Per esempio:
- Deadlock o race conditions per il software distribuito.
- Problemi dovuti al polimorfismo o al binding dinamico nel software object-oriented.
# References