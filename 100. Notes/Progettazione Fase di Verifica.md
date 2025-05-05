**Data time:** 13:52 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: [[Bachelor's Degree]]
# Progettazione Fase di Verifica

I progettisti della fase di verifica devono:
- scegliere e programmare la giusta combinazione di tecniche per raggiungere il livello richiesto di qualità entro i limiti di costo.
- Progettare una soluzione specifica che si adatta: al problema, ai requisiti e all'ambiente di sviluppo.
Tutto questo senza poter contare su "ricette" fisse.

**5 domande sa usare come guida**
1. <u>Quando iniziare verifica e convalida? Quando sono completo ?</u> 
	Il testing non è una fase finale dello sviluppo software. L'esecuzione dei test è solo una piccola parte del processo di verifica e convalida.
	
	V&V iniziano non appena decidiamo di creare un prodotto software. V&V durano molto oltre la consegna dei prodotti, di solito per tutto il tempo in cui software è in uso, per far fronte alle evoluzioni e agli adattamenti alle nuove condizione. 
	
	Quando iniziare quindi verifica e convalida?
	**Opzione 1**. Dallo studio di fattibilità di un nuovo progetto, essendo che deve tener conto delle qualità richieste e dell'impatto sul costo complessivo.
	
	In questa fase, le attività correlate alla qualità comprendono:
	- analisi del rischio.
	- definizione delle misure necessarie per valutare e controllare la qualità in ogni stadio di sviluppo.
	- valutazione dell'impatto di nuove funzionalità e nuovi requisiti di qualità.
	- valutazione economica delle attività di controllo della qualità: costi e tempi di sviluppo.
	
	**Opzione 2**. Dopo il rilascio. Le attività di manutenzione comprendono:
	- analisi delle modifiche ed estensioni.
	- generazione di nuove suite di test per le funzionalità aggiuntive.
	- riesecuzione dei test per verificare la non regressione delle funzionalità del software dopo le modifiche e le estensioni.
	- rilevamento e analisi dei guasti.

2. <u>Quali tecniche applicare ?</u>
	Nessuna singola tecnica di analisi e testing (A & T) è sufficiente per tutti gli scopi. 
	Le principali ragioni per combinare diverse tecniche sono: 
	- Efficacia per diverse classi di difetti: analisi statica invece di test per le race conditions.
	- Applicabilità in diverse fasi del processo di sviluppo, per esempio: ispezione per la convalida dei requisiti iniziali. 
	- Differenze negli scopi. Esempio: test statistico per misurare l'affidabilità.
	- Compromessi in termini di costo e affidabilità: usare tecniche costose solo per requisiti di sicurezza.
	
	![[Screenshot 2023-11-29 at 13.07.43.png]]
	
3. <u>Come possiamo valutare se un prodotto è pronto per essere rilasciato?</u>
	Alcune misure di **dependability**:
	- La **disponibilità** misura la qualità di un sistema in termini di tempo di esecuzione rispetto al tempo in cui il sistema è giù.
	- Il **tempo medio tra i guasti** (MTBF) misura la qualità di un sistema in termini di tempo tra un guasto ed il successivo.
	- **L'affidabilità** indica la percentuale di operazioni che terminano con successo.
	
	Per valutare se un prodotto è pronto per il rilascio dobbiamo definire bene le misura.
	
	**Esempio**. Applicazione e-shop realizzata con 100 operazioni. Il software funziona correttamente fino al punto in cui viene indicata una carta di credito: nel 50% dei casi viene addebitato l'importo sbagliato.

	Qual è l'affidabilità del sistema? Se contiamo la percentuale di operazioni corrette, solo una operazione su 100 fallisce: il sistema è affidabile al 99%. Se contiamo le sessioni, solo il 50% affidabile.
	
	**Alfa test**: test eseguiti dagli sviluppatori o dagli utenti in ambiente controllato, osservati dall'organizzazione dello sviluppo.
	**Beta test**: test eseguiti da utenti reali nel loro ambiente, eseguendo attività reali senza interferenze o monitoraggio ravvicinato.
	
4. <u>Come possiamo controllare la qualità delle release successive?</u>
	Attività dopo la consegna:
	- test e analisi del codice nuovo e modificato.
	- riesecuzione dei test di sistema.
	- memorizzazione di tutti i bug trovati.
	- test di regressione. Quasi automatico.
	- distinzione tra "major" e "minor" revisions: 2.0 vs 1.4, 1.5 vs 1.4.

5. <u>Come può essere migliorato il processo di sviluppo?</u>
	Si incontrano gli stessi difetti progetto dopo progetto:
	- identificare e rimuovere i punti deboli nel processo di sviluppo. Per esempio cattive pratiche di programmazione.
	- identificare e rimuovere i punti deboli del test e dell'analisi che consentono loro di non essere individuati

**Verifica vs Convalida**
*Definizione*: La **convalida** (aka validazione) risponde alla domanda: stiamo costruendo il sistema che serve all'utente? 

*Definizione*: La **verifica** risponde alla domanda: Stiamo costruendo un sistema che rispetta le specifiche?

![[Screenshot 2023-11-29 at 13.24.47.png]]

![[Screenshot 2023-11-29 at 13.24.59.png]]

**Terminologia IEEE: malfunzionamento**
Con un **Malfunzionamento** Il sistema software a tempo di esecuzione non si comporta secondo le specifiche. Es. output non atteso. Un malfunzionamento ha una natura dinamica: può essere osservato solo mediante esecuzione. Causato da un difetto (o più difetti).

**Terminologia IEEE: difetto**
Un **difetto** (o anomalia, bug, fault) è un difetto nel codice (appartiene alla struttura statica del programma), l'atto di correzione dagli difetti è detto debug o bugfixing.

Normalmente causa un malfunzionamento, ma non sempre, in questo caso si dice che il **difetto è latente**, ad esempio, il caso in cui il difetto è contenuto in un cammino che non viene praticamente mai eseguito; un altro caso è rappresentato dalla presenza di più difetti il cui effetto totale è nullo.

**Esempio**
```c
int raddoppia(int x) {
	return x * x;
}```

**Terminologia IEEE: errore**
Un **errore** è la cause di un difetto, può essere causato da un incomprensione umana nel tentativo di comprendere o risolvere un problema, o nell'uso di strumenti.

Per esempio nel metodo raddoppia, se c'è un difetto è errore di editing.

# References