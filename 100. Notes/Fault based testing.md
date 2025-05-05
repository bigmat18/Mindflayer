**Data time:** 14:26 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: [[Bachelor's Degree]]
# Fault based testing

Ipotizza dei difetti potenziali del codice sotto test. Crea e valuta una test suite sulla base della sua capacità di rilevare i difetti ipotizzati.

La più nota tecnica di fault based testing è il **test mutazionale** dove si iniettano difetti modificando il codice.

### Test mutazionale
Precondizione: aver esercitato un programma P su una batteria di test T, e aver verificato P corretto rispetto a T.
1. Si vuole fare una verifica più profonda sulla correttezza di P: si introducono dei difetti (piccoli, dette mutazioni) su P e si chiami P’ il programma modificato: P’ viene detto mutante.
2. Si eseguono su P’ gli stessi test di T. Il test dovrebbe manifestare dei malfunzionamenti. Se il test non rileva questi difetti, allora significa che la batteria di test non era abbastanza buona. Se li rivela, abbiamo una maggior fiducia sella batteria di test. 
Questo è un metodo per valutare la capacità di un test, e valutare se è il caso di introdurre test più sofisticati.

*Definizione* (**Mutazione**): una mutazione è una piccolo cambiamento in un programma.

**Esempio**. si cambia (i < 0) in (i <= 0).

**Ipotesi**: i difetti reali sono piccole variazioni sintattiche del programma corretto => mutanti sono modelli ragionevoli dei programmi con difetti.

**Che cos'è il test mutazionale?**
È un metodo di test strutturale volto a:
- Valutare/migliorare l'adeguatezza delle suite di test 
- Sistemare il numero di difetti nei sistemi sotto test.

Il processo dato un programma P e una suite di test T è il seguente:
1. Applichiamo delle mutazioni a P per ottenere una sequenza P1, P2, ..., Pn di mutanti di P.
2. Ogni mutante deriva dall'applicazioni di una singola operazione di mutazione a P.
3. Si esegue la suite di test T su ciascuno dei mutanti.

Si dice che **T uccide il mutante Pj** se rileva un malfunzionamento:
- il mutante viene ucciso se fallisce almeno in un caso di test di T
- Si dice anche che il caso di test in questione ha ucciso il mutente.

*Definizione* (**Efficacia di un test**) = quantità di mutanti uccisi/numero mutanti.
Se T uccide k mutanti su n, l'efficacia di T è k/n.

Un mutante sopravvive a una test suite se per tutti i test case della test suite non si distingue l’esito del test se eseguito sul programma originale o su quello mutante.

La tecnica si applica in congiunzione con altri criteri di test. Nella sua formulazione è prevista infatti l’esistenza, oltre al programma da controllare, anche di un insieme di test già realizzati. Uno dei vantaggi di questo approccio è che può essere quasi completamente automatizzato.

**Mettendo insieme tutte le ipotesi**
Test che trovano semplici difetti allora trovano anche difetti più complessi, una test suite che uccide i mutanti è capace anche di trovare difetti reali nel programma.

### Test mutazionale per valutare qualità di batteria di test
Specifica: la funzione foo restituisce x + y se x <= y e x * y altrimenti.
**Originale**
```java
int foo(int x, int y){
	if (x <= y) return x + y;
	else return x * y;
}
```
**Mutante**
```java
int foo(int x, int y){
	if (x < y) return x + y;
	else return x * y;
}
```

Consideriamo la batteria di test: { <(0,0), 0>, < (2,3), 5>, <(4,3), 12> }
Il mutante non viene ucciso (sopravvive), la batteria è poco adeguata e va ri-progettata anche se copre:
- criteri strutturali: tutte le decisioni, tutte le istruzioni.
- criteri funzionali: le classi di equivalenze e la frontiera.

### Test mutazione per stimare numero di difetti nel sistema
Immagina di dover contare il numero di pesci in un lago.
Mettiamo M pesci meccanici nel lago che contiene un numero imprecisato di pesci. Osserviamo N pesci e vediamo che di questi N, N1 sono meccanici.

Assunzione: i difetti che mettiamo sono rappresentativi di quelli che potrebbero esserci davvero

Ne avevamo messi M meccanici. Ne osserviamo N. Di questi N1 sono meccanici. N1: N = M : Total, allora: $$Total = \large\frac{N \times M}{N_1}$$
**Esempi di mutazioni**
- crp: sostituzione (replacement) di costante per costante. Ad esempio: da (x <5) a (x <12) 
- ror: sostituzione dell'operatore relazionale. Ad esempio: da (x <= 5) a (x <5) 
- vie: eliminazione dell'inizializzazione di una variabile. Cambia int x = 5; a int x.
- lrc: sostituzione di un operatore logico. Ad esempio da & a | 
- abs: inserimento di un valore assoluto. Da x a |x|

*Definizione* (**Mutanti validi/invalido**): un mutante è invalido se non è sintatticamente corretto cioè se non passa la compilazione, è valido altrimenti.

*Definizione* (**Mutante utile**): un mutante è utile se è valido e distinguerlo dal programma originale non è facile. Cioè esiste solo un piccolo sottoinsieme di casi di test che permette di distinguerlo dal programma originale.

*Definizione* (**Mutante inutile**): un mutante è inutile se è ucciso da quasi tutti i casi di test.

Trovare mutazioni che producano mutanti validi e utili non è facile e dipende dal linguaggio.

**Come sopravvive un mutante**
- Un mutante può essere **equivalente** al programma originale. Cambiare (x <=0) a (x < 0 OR x=0) non ha cambiato l'output: La mutazione non è un vero difetto. Determinare se un mutante è equivalente al programma originale può essere facile o difficile; nel peggiore dei casi è indecidibile.
- Oppure la suite di test potrebbe essere inadeguata. Se il mutante poteva essere stato ucciso, ma non lo era, indica una debolezza nella suite di test

**Esempi**![[Screenshot 2023-12-05 at 12.51.02.png]]

Questa strategia è adottata con obiettivi diversi:
- favorire la scoperta di malfunzionamenti ipotizzati: intervenire sul codice può essere più conveniente rispetto alla generazione di casi di test ad hoc. 
- valutare l’efficacia dell’insieme di test, controllando se “si accorge” delle modifiche introdotte sul programma originale. 
- cercare indicazioni circa la localizzazione dei difetti la cui esistenza è statadenunciata dai test eseguiti sul programma originale.

Uso limitato dal gran numero di mutanti che possono essere definiti, dal costo della loro realizzazione, e soprattutto dal tempo e dalle risorse necessarie a eseguire i test sui mutanti e a confrontare i risultati.
# References