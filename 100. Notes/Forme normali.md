**Data time:** 00:46 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Normalizzazione]]

**Area**: [[Bachelor's Degree]]
# Forme normali

Una forma normale è una proprietà di una base di dati relazionale che ne garantisce la “qualità”, cioè l'assenza di determinati difetti. Quando una relazione non è normalizzata:
- presenta ridondanze
- si presta a comportamenti poco desiderabili durante gli aggiornamenti

**1FN**: Impone una restrizione sul tipo di una relazione: ogni attributo ha un tipo elementare.
**2FN, 3FN**: Impongono restrizioni sulle dipendenze.
**FNBC**: FNBC (Boyce-Codd) è la più naturale e la più restrittiva.

Una relazione r è in forma normale di Boyce e Codd (BCNF) se, per ogni dipendenza funzionale (non banale) X → Y definita su di essa, X contiene una chiave K di r (è una superchiave).

La forma normale richiede che i concetti in una relazione siano omogenei (solo proprietà direttamente associate alla chiave).

**Esempio**. Relazione sugli impiegati
![[Screenshot 2023-12-08 at 12.39.51.png | 400]]
Non è in forma normale di Boyce and Codd perché esiste la dipendenza funzionale.
Impiegato → Stipendio (Impiegato non è (super)chiave per la relazione )

### FORME NORMALI di Boyce-Codd
**FNBC**: L'Intuizione è che se esiste in R una dipendenza X→A non banale ed X non è chiave, allora X modella l’identità di un’entità diversa da quelle modellate dall’intera R.

**Esempio**. StudentiEdEsami(<u>Matricola</u>, Nome, Provincia, AnnoNascita, <u>Materia</u>, Voto).
Matricola -> Nome e Matricola non è (super)chiave. Il Nome dipende dalla Matricola che non è chiave.

*Definizione*: R<T,F> è in BCNF $\Leftrightarrow$ per ogni $X \to A \in F^+$  con $A \notin X$ (non banale) X è una superchiave.

*Teorema*: R<T,F> è in BCNF $\Leftrightarrow$ per ogni $X \to A \in F$ non banale X è superchiave.

**Esempio**. 
Docenti(<u>CodiceFiscale</u>, Nome, Dipartimento, Indirizzo)    {CF → N D; D → I} 
Impiegati(<u>Codice</u>, Qualifica, <u>NomeFiglio</u>)                            {C → Q}

**Algoritmo di analisi**
R<T,F> è decomposta in $R_1(X,Y)$ e $R_2(X,Z)$ e su di esse si ripete il procedimento; esponenzialmente.

input: R<T, F> con F copertura canonia
output: decomposizione in BCNF che preserva i dati $\rho = \{R<T,F>\}$ 

while esiste in $\rho$ una $R_i<T_i, F_i>$ non in BCNF per la DF $X \to A$
do
	$T_a = X^+$
	$F_a = \pi_{T_a}(F_i)$
	$T_b = T_i - X^+ + X$ (Attenzione: errore nel vecchio libro)
	$F_b = \pi_{Tb}(F_i)$
	$\rho = \rho - R_i + \{R_a<T_a, F_a>, R_b<T_b, F_b>\}$    ($R_a$ ed $R_b$ sono nomi nuovi)
end

*Osservazione*: $T_b = T_i - X^+ + X$, perché aggiungiamo X?

**Preserva i dati** ma non necessariamente le dipendenze

**Esempio** di decomposizione senza perdita di dipendenze: 
Docenti(CodiceFiscale, Nome, Dipartimento, Indirizzo), {CF → N D; D → I}
(CF)+= CF N D I      è chiave 
(D) += D I                 non è chiave

Decompongono:
- R1(<u>D</u>, I)              R2(<u>CF</u>, D)
- F1 = $\{D \to I\}$    F2=$\{CF \to ND\}$

**Esempio**. Impiegati(Codice, Qualifica, NomeFiglio) {C → Q}
- R1(C, Q);            R2(C, NF) 
- F1 = { C → Q }    F2={ }

Vediamo che dato che non perdo dipendenze funzionali, posso fare la proiezione approssimata su F.

**Esempio**. Telefoni (Prefisso, Numero, Località, Abbonato, Via)
F = { P N → L A V, L → P }
- R1(L, P); R2(L, N, A, V) 
- Preserva dati ma non le dipendenze: PN → L non è deducibile da F1 e F2.

Cosa vuol dire che "non preserva le dipendenze"?
- R1 = {(Pisa, 050); (Calci, 050)}
- R2 = {(Pisa, 506070, Rossi, Piave), (Calci,506070, Bianchi, Isonzo)} 
Posso inserire due numeri telefonici (senza prefisso) con comuni differenti che hanno lo stesso prefisso?

**Esempio**.
Dato il seguente schema relazionale R<ABCDE, F= { CE→A, D→E, CB→E, CE→B }>. Applicare l’algoritmo di analisi e dire se dati e dipendenze sono stati preservati.
- Consideriamo CE→A. $CE^+$=CEAB (CE non è chiave), per cui decomponiamo:
	R1(CEAB) (gli attributi di $CE^+$), R2(CED) 
	(In R2 tutti gli altri attribuiti (D) e la chiave esterna (CE))
- Proiettiamo le dipendenze (approssimate su F): 
	R1 < CEAB, { CE → A, CB → E, CE → B } > (Proiezione in F) 
	R2 < CED, { D→E } > (Proiezione in F)
$CE^+$=CEAB e $CB^+$=CBEA, per cui R1 è in BCNF
$D^+$=DE, per cui R2 va ancora decomposta: R2 < CED, { D→E } > -> R3 , R4
La decomposizione è quindi: { R1(CBEA), R3(DE), R4(DC) }.
La decomposizione preserva dati e dipendenze ed è in questo caso è la stessa prodotta dall’algoritmo di sintesi (che vedremo dopo).
### Terza forma normale
Chiediamoci ora se data una relazione non in FNBC, è sempre possibile ottenere una decomposizione in FNBC?
![[Screenshot 2023-12-08 at 14.14.37.png | 450]]
- Progetto Sede → Dirigente: ogni progetto ha più dirigenti che ne sono responsabili, ma in sedi diverse, e ogni dirigente può essere responsabile di più progetti; però per ogni sede, un progetto ha un solo responsabile 
- Dirigente → Sede: ogni dirigente opera presso una sede 
- Dirigente → Sede è una dipendenza funzionale ma Dirigente non è una (super)chiave. Quindi la relazione non è in BCNF.
- Progetto Sede → Dirigente coinvolge tutti gli attributi e quindi nessuna decomposizione può preservare tale dipendenza. 

Quindi in alcuni casi la BCNF “non è raggiungibile”. Occorre ricorrere a una forma normale indebolita.
Quando si hanno diverse DF è difficile ragionare sullo schema, ed è quindi altrettanto difficile operare manualmente buone decomposizioni. La terza forma normale (3NF) è un target di normalizzazione che consente di ottenere automaticamente: 
- decomposizioni senza perdita
- decomposizioni che preservano tutte le dipendenze

*Definizione*: Una relazione r è in terza forma normale (3NF) se, per ogni FD (non banale) X → Y definita su r, è verificata almeno una delle seguenti condizioni:
- X contiene una chiave K di r (come nella BCNF)
- Oppure ogni attributo in Y è contenuto in almeno una chiave K di r

Prendendo l'esempio di prima. 
- Progetto Sede → Dirigente 
- Dirigente → Sede
Nella prima dipendenza funzionale il primo membro della dipendenza (Progetto, Sede) è una chiave, nella seconda il secondo membro (Sede) è contenuto in una chiave. Quindi la relazione è in terza forma normale.

**Svantaggi**
La 3FN è **meno restrittiva** della FNBC:
- Tollera alcune ridondanze ed anomalie sui dati. Es. per ogni occorrenza di un dirigente viene ripetuta la sua sede.
- Certifica meno lo qualità dello schema ottenuto.

**Vantaggi**
La 3FN è sempre ottenibile, qualsiasi sia lo schema di partenza. COME? Algoritmo di normalizzazione in TFN!

*Definizione*: $R<T,F>$ è in 3FN se per ogni $X \to A \in F^+$ con $A \notin X$ X è superchiave o A è primo.

Nota: 
- La 3FN ammette una dipendenza non banale e non-dachiave se gli attributi a destra sono primi; 
- la BCNF non ammette mai nessuna dipendenza non banale e non-da-chiave.

*Teorema*: R<T,F> è in 3FN se per ogni $X \to A \in F$ non banale, allora X è una superchiave oppure A è primo.

Non sono in 3FN (e quindi, neppure in BCNF).
- Docenti(CodiceFiscale, Nome, Dipartimento, Indirizzo)     { CF → N D; D → I }
- Impiegati(Codice, Qualifica, NomeFiglio)                             { C → Q }
- Telefoni(Prefisso, Numero, Località, Abbonato, Via)           {P N → L A V, L → P} 
	Chiavi = {PN, LN}
- Esami(Matricola, Telefono, Materia, Voto) 
	Matricola Materia → Voto, Matricola → Telefono, Telefono → Matricola
	Chiavi: Matricola Materia, Telefono Materia

### Algoritmo di sintesi
Input: Un insieme R di attributi e un insieme F di dipendenze su R.
Output: Una decomposizione $\rho = \{S_i\}_{i=1..n}$ di R tale che preservi dati e dipendenze e ogni $S_i$ sia in 3NF, rispetto alle proiezioni di F su $S_i$ .
begin:
	<u>Passo 1</u>: Trova una copertura canonica G di F e poni $\rho = \{\}$
	<u>Passo 2</u>: Sostituisci in G ogni insieme $X \to A_1 , \dots , X \to A_h$ di dipendenze con lo stesso determinante, con la dipendenza $X \to A_1$・ ・ ・ $A_h$.
	<u>Passo 3</u>: Per ogni dipendenza X → Y in G, metti uno schema con attributi XY in $\rho$.
	<u>Passo 4</u>: Elimina ogni schema di $\rho$ contenuto in un altro schema di $\rho$.
	<u>Passo 5</u>: Se la decomposizione non contiene alcuno schema i cui attributi costituiscano una superchiave per R, aggiungi ad essa lo schema con attributi W, con W una chiave di R.
end

**Esempio**. Dati R di attributi, ed un insieme di dipendenze F, l’algoritmo di sintesi di schemi in terza forma normale procede come segue.
Impiegato(Matricola, Cognome, Grado, Retribuzione, Dipartimento, Supervisore, Progetto, Anzianità)
{M → RSDG, MS → CD, G → R, D →S, S → D, MPD → AM}
- <u>Passo 1</u> Costruire una copertura canonica G di F. 
	F={M→RSDG, MS → CD, G →R, D→S, S→D, MPD→AM}
	G={M→D, M→G, M →C, G→R, D→S, S→D, MP→A}
- <u>Passo 2a</u> Decomporre G nei sottoinsiemi $G^{(1)}, G^{(2)},\dots, G^{(n)}$ , tali che ad ogni sottoinsieme appartengano dipendenze con gli stessi lati sinistri (facoltativo)
	- $G^{(1)}$ = {M→D, M →G, M→C}
	- $G^{(2)}$ = {G→R}
	- $G^{(3)}$ = {D→S}
	- $G^{(4)}$ = {S→D}
	- $G^{(5)}$ = {MP→A}
- <u>Passo 2b</u> sostituisci in G ogni insieme X → $A_1$, …, X→ $A_h$ di dipendenze con lo stesso determinante, con la dipendenza X → $A_1, \dots, A_h$.
	- $G^{(1)}$ = {M→D, M →G, M→C}             $G^{(1)}$ = {M→DGC}
	- $G^{(2)}$ = {G→R}                                     $G^{(2)}$ = {G→R}
	- $G^{(3)}$ = {D→S}                  ==>            $G^{(3)}$ = {D→S}
	- $G^{(4)}$ = {S→D}                                     $G^{(4)}$ = {S→D}
	- $G^{(5)}$ = {MP→A}                                  $G^{(5)}$ = {MP→A}
- <u>Passo 3</u>. Trasformare ciascun $G^{(i)}$ in una relazione $R^{(i)}$ con gli attributi contenuti in ciascuna dipendenza. Il lato sinistro diventa la chiave della relazione:
	 Passo 2                                                Passo 3
	- $G^{(1)}$ = {M→DGC}                               $R^{(1)}$ = {<u>M</u>DGC}
	- $G^{(2)}$ = {G→R}                                     $R^{(2)}$ = {<u>G</u>R}
	- $G^{(3)}$ = {D→S}                  ==>            $R^{(3)}$ = {<u>DS</u>}
	- $G^{(4)}$ = {S→D}                                     $R^{(4)}$ = {<u>SD</u>}
	- $G^{(5)}$ = {MP→A}                                  $R^{(5)}$ = {<u>MP</u>A}
- <u>Passo 4</u>. Si eliminano schemi contenuti in altri. 
![[Screenshot 2023-12-08 at 15.14.52.png | 500]]
- <u>Passo 5</u>. Se nessuna relazione $R^{(i)}$ così ottenuta contiene una (super)chiave K di R(U), inserire una nuova relazione $R^{(n+1)}$ contenente gli attributi della chiave.
	Impiegato (Matricola, Cognome, Grado, Retribuzione, Dipartimento, Supervisore, Progetto, Anzianità) 
	F={M→RSDG, MS → CD, G →R, D→S, S→D, MPD→AM} 
	G={M→D, M→G, M →C, G→R, D→S, S→D, MP→A}
	a chiave è costituita da: (MP). 
	Dallo step 4: $R^{(1)}$(MDGC) $R^{(2)}$(GR) $R^{(3)}$(SD) $R^{(5)}$(MPA)
	$R^{(5)}$(MPA) contiene la chiave → non c’è necessità di aggiungere altre relazioni!

In conclusione, data la relazione: R(MGCRDSPA), con dipendenze funzionali: 
Impiegato (Matricola, Cognome, Grado, Retribuzione, Dipartimento, Supervisore, Progetto, Anzianità) 
F= {M→RSDG, MS → CD, G →R, D→S, S→D, MPD→AM} 
G={M→D, M→G, M →C, G→R, D→S, S→D, MP→A}

La sua decomposizione in 3FN è la seguente:
$R^{(1)}$(MDGC)    $R^{(2)}$(GR)     $R^{(3)}$(SD)     $R^{(54}$(MPA)


### Dipendenze multi-valore
![[Screenshot 2023-12-08 at 15.22.34.png | 500]]

La coesistenza di due proprietà multivalore INDIPENDENTI, fa sì che per ogni impiegato esistono tante ennuple quante sono le possibili coppie di valori di Qualifica e NomeFiglio.

Tre casi di relazioni con proprietà multivalori. Si possono risolvere usando le decomposizioni?
La coesistenza di due proprietà multivalore indipendenti, fa sì che per ogni impiegato esistano tante ennuple quante sono le possibili coppie di valori di Stipendio e NomeFiglio.

Decomponendo lo schema in due sottoschemi in modo da modellare separatamente le proprietà multivalori indipendenti, si avrebbe una base di dati priva di anomalie:
- StipendiImpiegati(Codice, StoriaStipendio) 
- FigliImpiegati(Codice, NomeFiglio)
# References