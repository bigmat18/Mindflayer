**Data time:** 00:46 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Normalizzazione]]

**Area**: [[Bachelor's Degree]]
# Decomposizione di schemi

In generale, per eliminare anomalie da uno schema occorre decomporlo in schemi più piccoli "equivalenti"

**Esempi** di decomposizione.
L’intuizione è che si devono “estrarre” gli attributi che sono determinati da attributi non chiave ovvero “creare uno schema per ogni funzione”.

![[Screenshot 2023-12-07 at 16.41.44.png]]

La soluzione non è tuttavia sempre così semplice, bisogna fare anche altre considerazioni; ad esempio, operando come prima:
![[Screenshot 2023-12-07 at 16.51.50.png | 500]]

**Decomposizione sulla base delle dipendenze**
![[Screenshot 2023-12-07 at 16.52.37.png | 500]]

**Proviamo a ricostruire mediante join**
![[Screenshot 2023-12-07 at 16.52.58.png | 500]]

*Definizione*: Dato uno schema R(T), $p = \{R_1(T_1), \dots, R_k(T_k)\}$ è una **decomposizione** di R sse $T_1 \cup \dots \cup T_k = T$.

**Esempio**. StudentiEdEsami(Matricola, Nome, Provincia, AnnoNascita, Materia, Voto)
- {Studenti(Matricola, Nome, Provincia,AnnoNascita), Esami(Matricola, Materia, Voto)}
- {Studenti(Matricola, Nome, Provincia,AnnoNascita), Esami(Nome, Materia, Voto)} 
- {Studenti(Matricola, Nome, Provincia,AnnoNascita), Esami(Materia, Voto)}

In generale, per eliminare anomalie da uno schema occorre decomporlo in schemi più piccoli "equivalenti"

Due proprietà desiderabili di una decomposizione: 
- **conservazione dei dati** (nozione semantica) 
- **conservazione delle dipendenze**

### Preservare i dati
*Definizione*: $p = \{R_1(T_1), \dots, R_k(T_k)\}$ è una decomposizione di uno schema R(T) che **preserva i dati** sse per ogni istanza valida r di R: 
$$r = (\pi_{T1}r) \lor (\pi_{T2}r) \lor \dots \lor (\pi_{Tk}r)$$
Dalla definizione di giunzione naturale scaturisce il seguente risultato:
*Teorema*: $p = \{R_1(T_1), \dots, R_k(T_k)\}$ è una decomposizione di R(T) allora per ogni istanza r di R: $r \subseteq (\pi_{T1}r) \lor (\pi_{T2}r) \lor \dots \lor (\pi_{Tk}r)$ 

**Decomposizione con perdita di informazioni**
Prendiamo il seguente esempio: 
StudentiEdEsami(Matricola, Nome, Provincia, AnnoNascita, Materia, Voto) ==>
{Studenti(Matricola, Nome, Provincia,AnnoNascita), Esami(Nome, Materia, Voto)}

Cosa succede quando si fa la giunzione? Nessuna tupla si perde, ma…?
Questa decomposizione crea tuple spurie: ci sono n-uple in più. Si pensi al caso di due persone con lo stesso nome che hanno sostenuto esami diversi, cosa succede dopo la giunzione? Perdita di informazione!

Mentre con:
{Studenti(Matricola, Nome, Provincia, AnnoNascita), Esami(Matricola, Materia, Voto)}
Non si perdono informazioni perché la chiave è l’unico modo per avere una decomposizione senza perdita di informazione.

**Decomposizione senza perdita**
Uno schema R(X) si decompone senza Perdita dei dati negli schemi R1(X1) ed R2(X2) se, per ogni possibile istanza r di R(X), il join naturale delle proiezioni di r su X1 ed X2 produce la tabella di partenza. (cioè non contiene ennuple spurie)
$$\pi_{X1}(r) \bowtie \pi_{X2}(r) = r$$
La decomposizione senza perdita è garantita se l’insieme degli attributi comuni alle due relazioni ($X_1 \cap X_2$) è chiave per almeno una delle due relazioni decomposte. 
Ad esempio, Sede=(Progetto, Sede)$\cap$(Impiegato, Sede) non è chiave per nessuna delle due relazioni
![[Screenshot 2023-12-07 at 17.06.12.png | 400]]

*Definizione* (**Senza perdita**): Sia r una relazione su un insieme di attributi X e siano $X_1$ e $X_2$ due sottoinsiemi di X la cui unione sia pari a X stesso; Inoltre, sia $X_0$ l’intersezione di $X_1$ e $X_2$ , allora: 
- r si decompone senza perdita su $X_1$ e $X_2$ se soddisfa la dipendenza funzionale $X_0 \to X_1$ oppure la dipendenza funzionale $X_0 \to X_2$.

*Teorema (non formale)*: Se l’insieme degli attributi comuni alle due relazioni $(X_1 \cap X_2)$ è chiave per almeno una delle due relazioni decomposte allora la decomposizione è senza perdita.

*Dimostrazione (non formale)*:
- Supponiamo r sia una relazione sugli attributi ABC e consideriamo le sue proiezioni $r_1$ su AB e $r_2$ su AC. 
- Supponiamo che r soddisfi la dipendenza funzionale A → C. Allora A è chiave per r su AC e quindi non ci sono in tale proiezione due tuple diverse sugli stessi valori di A.
Il join costruisce tuple a partire dalle tuple nelle due proiezioni: Sia t=(a,b,c) una tupla del join di $r_1$ e $r_2$. Mostriamo che appartiene ad r (cioè non è spuria):
- t è ottenuta mediante join da t1=(a,b) di r1 e t2=(a,c) su r2
- Allora per la definizione di proiezione, esistono due tuple in r, $t’_1$= (a,b,$*$) e $t'_2$= (a,$*$,c) (dove $*$ sta per un valore non noto).
- Poiché A →C, allora esiste un solo valore in C associato al valore a. Dato che (a,c) compare nella proiezione, questo valore è proprio c.
- Ma allora nella tupla $t’_1$ il valore incognito deve essere proprio c. Quindi (a,b,c) appartiene a r.

**Decomposizioni binarie**
*Teorema*: Sia R<T, F> uno schema di relazione, la decomposizione $p = \{R_1(T_1), \dots, R_k(T_k)\}$ preserva i dati sse:
$$T_1 \cap T_2 \to T_1 \in F^+ \:oppure\: T_1 \cap T_2 \to T_2 \in F^+$$
Esistono criteri anche per decomposizioni in più di due schemi.

Anche se una decomposizione è senza perdite, può comunque presentare dei problemi di conservazione delle dipendenze.

**Esempio**. Impiegato=(Impiegato, Sede)$\cap$(Impiegato, Progetto)
![[Screenshot 2023-12-07 at 17.16.52.png | 500]]
Con questa decomposizione, non ho tuple spurie

![[Screenshot 2023-12-07 at 17.17.32.png | 500]]
In questa decomposizione: trascuriamo la seconda dipendenza funzionale.

Supponiamo di voler inserire una nuova ennupla che specifica la partecipazione dell'impiegato Neri (che opera a Milano) al progetto Marte (lo schema non lo impedisce).
![[Screenshot 2023-12-07 at 17.18.28.png]]
Viene violata la seconda dipendenza funzionale (che per il momento avevamo tenuto in sospeso) Progetto → Sede.
![[Screenshot 2023-12-07 at 17.19.25.png]]
Una decomposizione conserva le dipendenze se ciascuna delle dipendenze funzionali dello schema originario coinvolge attributi che compaiono tutti insieme in uno degli schemi decomposti Nell’esempio considerato Progetto → Sede non è conservata.
![[Screenshot 2023-12-07 at 17.19.36.png | 400]]

**Esempio**. Query di verifica.
Se una DF non si preserva diventa più complicato capire quali sono le modifiche del DB che non violano la FD stessa.
In generale si devono prima eseguire query SQL di verifica (o, meglio, fare uso di trigger)
Bisogna verificare che il progetto (Marte) sia presso la stessa sede dell’impiegato (Neri). A tal fine bisogna trovare un impiegato che lavora al progetto Marte.

![[Screenshot 2023-12-07 at 17.20.36.png]]

Una decomposizione: 
- deve essere senza perdita, per garantire la ricostruzione delle informazioni originarie 
- dovrebbe preservare le dipendenze, per semplificare il mantenimento dei vincoli di integrità originari.
Nell’esempio, questo suggerisce di inserire anche: va sempre eseguita una query, ma più semplice:
```sql
SELECT * -- OK se restituisce una tupla 
FROM Impiegati I, Progetti P 
WHERE I.Impiegato = ‘Neri’ AND P.Progetto = `Marte’ AND I.Sede = P.Sede
```

### Conservazione delle dipendenze
Una decomposizione **conserva le dipendenze** se ciascuna delle dipendenze funzionali dello schema originario coinvolge attributi che compaiono tutti insieme in uno degli schemi decomposti. 
Nell’esempio considerato Progetto → Sede non è conservata

**Esempio**
Telefoni(Prefisso, Numero, Località, Abbonato, Via)
{Pref Num → Loc Abb Via, Loc → Pref} cioè: {P N → L A V, L → P}
Si considera la decomposizione: 
ρ = {Tel <{N, L, A, V}, F1 >, Pref<{L, P}, F2 >} con F1 = {LN → A V} e F2 ={L → P} 

![[Screenshot 2023-12-08 at 01.44.16.png | 450]]
Preserva dati ma non le dipendenze: PN → L non è deducibile da F1 e F2.
Esistono istanze valide della decomposizione che non sono proiezione di una istanza valida della relazione originale.

*Definizione*: Dato lo schema $R<T, F>$ e $T_1 \subseteq T$ la **proiezione di F su $T_1$** è:
$$\pi_{T1}(F) = \{X \to Y \in F^+ \:|\: XY \subseteq T_1\}$$
**Esempio**. Sia R(A, B, C) e F = $\{A \to B, B \to C, C\to A\}$
$\pi_{AB}(F) \equiv \{A \to B, B \to A\}$
$\pi_{AC} \equiv \{A \to C, C \to A\}$
Potrebbe sembrare che la decomposizione $\rho_i = \{R_1(A, B), R_2(A,C)\}$ non preservi le dipendenze perché B e C non appaiono insieme in uno schema della decomposizione, invece da B → A e A → C si ha B→ C.

Algoritmo banale per il calcolo di $\pi_{T1}(F)$:
for each $Y \subseteq T_1$ do 
	$(Z\: :=\: Y^+; output \: Y \to Z \cap T_1)$

*Definizione*: Dato lo schema R<T, F> la decomposizione $\rho = \{R_1, \dots, R_n\}$ **preserva le dipendenze** sse l'unione delle dipendenze in $\pi_{Ti}(F)$ è una copertura di F.

*Proposizione*: Dato lo schema $R<T, F>$ il problema di stabilire se la decomposizione $\rho = \{R_1, \dots, R_n\}$ preserva le dipendenze ha la complessità tempo polinimiale.

*Teorema*: sia $p = \{R_i<T_i, F_i>\}$ una decomposizione R<T,F> che preserva le dipendenze e tale che $T_j$ per qualche j, è una superchiave per $R<T,F>$  allora $\rho$ preserva i dati.

**Esempio**
Telefoni(Prefisso, Numero, Località, Abbonato, Via) F={P N → L A V, L → P}
Si considera la decomposizione: $\rho = \{Tel<\{N, L, A, V\}>, Pref<\{L, P\}>, F2\}$ con
F1 = {LN → A V}    F2 = {L → P}

Preserva dati ma non le dipendenze: PN → L non è deducibile da F1 e F2.

![[Screenshot 2023-12-08 at 12.34.23.png]]
Una decomposizione dovrebbe sempre soddisfare le seguenti proprietà:
- la **decomposizione senza perdita**, che garantisce la ricostruzione delle informazioni originarie senza generazione di tuple spurie.
- la **conservazione delle dipendenze**, che garantisce il mantenimento dei vincoli di integrità (di dipendenza funzionale) originari.

# References