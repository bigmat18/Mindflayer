**Data time:** 00:45 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Normalizzazione]]

**Area**: [[Bachelor's Degree]]
# Dipendenze Funzionali

Per formalizzare la nozione di schema senza anomalie, occorre una descrizione formale della semantica dei fatti rappresentati in uno schema relazionale.

**Istanza valida di R**: è una nozione semantica, che dipende da ciò che sappiamo del dominio del discorso (non estensionale, non deducibile da alcune istanze dello schema).

Istanza valida r su R(T). Siano X, Y due sottoinsiemi non vuoti di T, esiste in r una **dipendenza funzionale** da X a Y se, per ogni coppia di ennuple $t_1, t_2$ di r con gli stessi valori su X, risulta che $t_1, t_2$ hanno gli stessi valori anche su Y.

La dipendenza funzionale da X a Y si denota con $X \rightarrow Y$.
**Esempio**
Persone (CodiceFiscale, Cognome, Nome, DataNascita).
CodiceFiscale $\rightarrow$ Cognome, Nome.

**Dipendenza funzionale vs chiave**
Istanza valida r su R(T) - Siano X e Y due sottoinsiemi non vuoti di T, esiste in r una dipendenza funzionale da X a Y (X→Y) se, per ogni coppia di ennuple t1 e t2 di r con gli stessi valori su X, risulta che t1 e t2 hanno gli stessi valori anche su Y.

**Esempio**
StudentiEdEsami(Matricola, Nome, Provincia,AnnoNascita, Materia, Voto)
Matricola → Nome, Provincia, AnnoNascita

*Definizione* (**Dipendenza funzionale**): Dato uno schema R(T) e X, $Y \subseteq T$, una dipendenza funzionale ( DF ) fra gli attributi X e Y, è un vincolo su R sulle istanze della relazione, espresso nella forma: $X \to Y$ 

i.e. X determina funzionalmente Y o Y è determinato da X, se per ogni istanza valida r di R un valore di X determina in modo univoco un valore di Y: $\forall$ r istanza valida di R:
$$\forall t1, t2 \in r \:se\: t1[X] = t2[X] \Rightarrow t1[Y] = t2[Y]$$
In altre parole: un’istanza r di R(T) soddisfa la dipendenza X → Y, (o X → Y vale in r), se per ogni coppia di ennuple t1 e t2 di r, se t1[X] = t2[X] allora t1[Y] = t2[Y].

**Esempio**
$\forall$ r istanza valida di R $\forall t1, t2 \in r \:se\: t1[X] = t2[X] \Rightarrow t1[Y] = t2[Y]$
Questa tabella soddisfa la dipendenza funzionale Matricola $\rightarrow$ Cognome.
![[Screenshot 2023-12-07 at 01.19.42.png]]

Si dice anche che:
- un instnza $r_0$ di R **soddisfa la DF** $X \rightarrow Y (r_0 |= X \rightarrow Y)$ se la proprietà vale per $r_0$ $\forall t1, t2 \in r \:se\: t1[X] = t2[X] \Rightarrow t1[Y] = t2[Y]$
- e che un istanza $r_0$ di R soddisfa un insieme F di DF se per ogni $X \rightarrow Y \in F$ vale che $r_0 |= X \rightarrow Y$ e questo se e solo se $\forall t1, t2 \in r \:se\: t1[X] = t2[X] \Rightarrow t1[Y] = t2[Y]$

**Esempio**
![[Screenshot 2023-12-07 at 01.23.42.png]]

Abbiamo usato un'unica relazione per rappresentare informazioni eterogenee:
- gli impiegati con i relativi stipendi (Impiegato → Stipendio)
- i progetti con i relativi bilanci (Progetto → Bilancio)
- le partecipazioni degli impiegati ai progetti con le relative funzioni (Impiegato Progetto → Funzione).

**Dipendente funzionali atomiche**
Ogni dipendenza funzionale X → A1 A2 …An , si può scomporre nelle dipendenze funzionali X → A1 , X → A2 , … , X → An Le dipendenze funzionali del tipo X → A si chiamano **dipendenze funzionali atomiche**.

**Esempio**. DotazioniLibri(CodiceLibro, NomeNegozio, IndNegozio, Titolo, Quantità)
CodiceLibro, NomeNegozio → IndNegozio, Titolo, Quantità ==>
CodiceLibro, NomeNegozio → IndNegozio
CodiceLibro, NomeNegozio → Titolo 
CodiceLibro,NomeNegozio → Quantità

**Esempio**
![[Screenshot 2023-12-07 at 01.27.30.png]]
PrezzoTot è il Prezzo di vendita Assumiamo che: Il tipo si riferisca ad una sola componente. Quali sono le dipendenze funzionali? Chiave: Kit, Tipo

Ridondanze:
- PrezzoTot è ripetuto in ogni tupla che si riferisce allo stesso kit.
- PrezzoComp è ripetuto in ogni tupla che ha lo stesso valore di Tipo e Fornitore
- Componente è ripetuto in ogni tupla che ha lo stesso Tipo

Quali sono le dipendenze funzionali? 
- Tipo → Componente 
- Kit → PrezzoTot 
- Kit,Tipo → PrezzoComponente, QuantComp, Fornitore

*Definizione* (**Decomposizione**): Una decomposizione della relazione che non presenti ridondaze e senza perdita di informazione.
![[Screenshot 2023-12-07 at 01.26.51.png]]
Dipendenze funzionali:
- Tipo → Componente
- Kit → PrezzoTot
- Kit,Tipo → PrezzoComponente, QuantComp, Fornitore

![[Screenshot 2023-12-07 at 01.29.40.png]]
*Definizione* (**Dipendenze banali e non**): La dipendenza funzionale del tipo ImpiegatoProgetto $\rightarrow$ Progetto è sempre valida per cui si tratta di un DF "banale". X → A è non banale se A non è contenuta in X.

Siamo interessati alle dipendenze funzionali non banali.

### Esprimere le dipendenze funzionali
Consideriamo: NomeNegozio → IndNegozio.

**Espressione diretta (P ⇒ Q):** se in due righe il NomeNegozio è uguale, anche l’IndNegozio è uguale: NomeNegozio${}_= \Rightarrow$ IndNegozio${}_=$

**Per contrapposizione (¬Q ⇒ ¬P):**  se l’IndNegozio è diverso allora il NomeNegozio è diverso:   NomeNegozio${}_\neq \Rightarrow$ IndNegozio${}_\neq$.

Per assurdo: non possono esserci due nuple in DotazioniLibri con NomeNegozio uguale e IndNegozio diverso: 
- Not (NomeNegozio${}_= \land$ IndNegozio${}_\neq$ ) 
- NomeNegozio${}_= \land$ IndNegozio${}_\neq \Rightarrow$ False

Sono **equivalenti**:
- NomeNegozio${}_= \Rightarrow$ IndNegozio${}_=$
- NomeNegozio${}_\neq \Rightarrow$ IndNegozio${}_\neq$
- NomeNegozio${}_= \Rightarrow$ IndNegozio${}_\neq \Rightarrow$ False

In generale: $A \Rightarrow B \Leftrightarrow A \land \lnot B \Rightarrow False \Leftrightarrow \lnot B \Rightarrow \lnot A$
Più in generale, in ogni clausola $A \land B \Rightarrow E \lor F$ posso spostare le sottoformule da un lato all’altro, negandole. Quindi sono equivalenti: 
- NomeNegozio$_=\: \land$ CodiceLibro$_= \Rightarrow$ Quantità$_=$ 
- NomeNegozio$_=\: \land$ CodiceLibro$_= \land$ Quantità$_\neq \Rightarrow$  False 
- CodiceLibro$_= \: \land$ Quantità$_\neq \Rightarrow$ NomeNegozio$_\neq$
- NomeNegozio$_= \: \land$ Quantità$_\neq \Rightarrow$ CodiceLibro$_\neq$

**Esempio**
Orari(CodAula, NomeAula, Piano, Posti, Materia, CDL, Docente, Giorno, OraInizio, OraFine)
1. In un dato momento, un docente si trova al più in un’aula 
2. Non è possibile che due docenti diversi siano nella stessa aula contemporaneamente 
3. Se due lezioni si svolgono su due piani diversi appartengono a due corsi di laurea diversi 
4. Se due lezioni diverse si svolgono lo stesso giorno per la stessa materia, appartengono a due CDL diversi.
    (lezioni diverse:  not(CodAula$_= \land$ and NomeAula$_=\land \dots$))

<u>Domanda 1</u>
![[Screenshot 2023-12-07 at 01.44.03.png]]

<u>Domanda 2</u>
![[Screenshot 2023-12-07 at 01.44.33.png]]

<u>Domanda 3</u>
![[Screenshot 2023-12-07 at 01.48.34.png]]

<u>Domanda 4</u>
![[Screenshot 2023-12-07 at 01.49.02.png]]

*Definizione*: R<T, F> denota uno **schema** con attributi T e dipendenze funzionali F.

Le DF sono una proprietà semantica, cioè dipendono dai fatti rappresentati e non da come gli attributi sono combinati in schemi di relazione. 

*Definizione* (**DF completa**): Si parla di **DF completa** quando X → Y e per ogni W$\subset$ X, non vale W → Y. 

Se X è una superchiave, allora X determina ogni altro attributo della relazione: X → T Se X è una chiave, allora X → T è una DF completa.

*Proprietà*: Da un insieme F di DF, in generale altre DF sono ‘implicate’ da F.

*Definizione* (**Dipendenze implicate**): Sia F un insieme di DF sullo schema R, diremo che F implica logicamente X → Y (F |= X → Y, ), se ogni istanza r di R che soddisfa F soddisfa anche X → Y.

*Definizione* (**Dipendenze banali**): implicate dal vuoto, es. {} |= X -> X.

**Esempio**
Sia r un’istanza di R, con F = {X → Y, X → Z} e X, Y, Z $\subseteq$ T. Sia X’ $\subseteq$ X. Altre DF sono soddisfatte da r, ad es.
- X $\rightarrow$ X' (DB banale) e
- X $\rightarrow$ YZ, infatti
    $t_1[X] = t_2[X] \Rightarrow t_1[Y] = t_2[Y]$
    $t_1[X] = t_2[X] \Rightarrow t_1[Z] = t_2[Z]$
    $t_1[X] = t_2[X] \Rightarrow t_1[YZ] = t_2[YZ]$
    Pertanto X → Y, Y → Z} |= X → Z
**Nota**: |= denota l'applicazione logica

**Regole di inferenza**
Come derivare DF implicate logicamente da F? Usando un insieme di regole di inferenza.

"**Assiomi**" (sono in realtà regole di inferenza) di **Armstrong**:
![[Screenshot 2023-12-07 at 02.03.25.png | 400]]

### [[Derivazione]]

### [[Chiusura di un Insieme F]]
# References