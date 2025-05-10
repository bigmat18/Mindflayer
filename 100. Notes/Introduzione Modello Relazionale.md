**Data time:** 16:00 - 10-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Modello Relazionale]]

**Area**: [[Bachelor's Degree]]
# Introduzione Modello Relazionale

Proposto da E. F. Codd nel 1970 per favorire l’indipendenza dei dati. Disponibile in DBMS reali nel 1981 (non è facile implementare l’indipendenza con efficienza e affidabilità!), si basa sul concetto matematico di relazione (con una variante), le relazioni hanno naturale rappresentazione per mezzo di tabelle.

*Definizione* (**Relazioni matematiche**): come nella teoria degli insiemi.
$$D_1,\: \dots, D_n \:\:\ \text(n \: insiemi \: anche \: non \: distinti)$$
*Definizione* **(Prodotto cartesiano)**: $D_1\: \times\: \dots\: \times \: D_n$  è l'insieme di tutte le n-uple $(d_1, \dots, d_n)$ tali che $d_1 \in D_1, \dots, d_n \in D_n$.

*Definizione* **Relazione matematica** su $D_1, \dots, D_n$: un sottoinsieme $D_1 \: \times \: \dots \: \times \: D_n$ di n-uple ordinate:
$$ (d_1, \dots, d_n) \:\:t.c.\:\: d_1 \in D_1, \dots, d_n \in D_n$$
$D_1, \dots, D_n$ sono i **domini** della relazione.

*Osservazione*: una relazioni è un insieme; quindi: non c'è ordinamento fra le n-uple, le n-uple sono distine, ciascun n-upla è ordinata, l'i-esimo valore proviene dall'i-esimo dominio.

**Esempio**
![[Screenshot 2023-11-22 at 16.47.46.png]]

Ciascuno dei domini ha due ruoli diversi, distinguibili attraverso la posizione: la struttura è posizionale

A ciascun dominio si associa un nome (**attributo**), che ne descrive il "ruolo"
Una tupla su un insieme di attributi $X$ è una funzione $t$ che associa a ciascun attributo $A$ un valore del suo dominio. Una relazione su $X$ è un insieme di tuple su $X$.

In un database una tabella rappresenta una relazione se:
- I valori di ogni colonna sono fra loro omogenei
- Le righe sono diverse fra loro
- Le intestazione delle colonne sono diverse tra loro

In una tabella che rappresenta una relazione l'ordinamento tra le righe è irrilevante ed anche l'ordinamento tra le colonne è irilevante.

Il modello relazionale è basato su valori. Ciò significa che i riferimenti fra dati in relazioni diverse sono rappresentati per mezzo di valori dei domini che compaiono nelle ennuple.

In ogni base di dati si distinguono
- lo **schema**, sostanzialmente invariante nel tempo, che ne descrive la struttura (aspetto intensionale). Le intestazioni delle tabelle.
- l'**istanza**, i valori attuali che possono cambiare anche molto rapidamente (aspetto estensionale). Il "corpo" di ciascuna tabella

![[Screenshot 2023-11-22 at 18.16.51.png]]

*Definizione*: I meccanismi per definire una base di dati con il modello relazione sono l'ennupla e la relazione:
- Un tipo ennupla $T$ è un insieme finito di coppie (Attributo, Tipo elemento)
- Se $T$ è un tipo ennupla, R(T) è lo schema della relazione $R$.
- Lo schema di una base di dati è un insieme di schemi di relaizoni $R_i(T_i)$.
- Un'istanza di uno schema $R(T)$ è un insieme finito di ennuple di tipo $T$.
Uguaglianza di due tipi ennupla, due ennuple, due tipi relazione.

I vantaggi principali della struttura basata su valori sono che:
- C'è un indipendenza dalla strutture fisiche (si potrebbe avere anche con puntatori di alto livello) che possono cambiare dinamicamente. La rappresentazione logica dei dati (costituita da soli valori) non fa riferimento a quella fisica.
- Si rappresenta solo ciò che è rilevante da punto di vista dell'applicazione.
- I dati sono portabili più facilmente da un sistema ad un altro.
- I puntatori sono direzionali.

*Definizione* (**schema di relazione**): un nome $R$ con un insieme $T$ di attribuiti $A_1, \dots, A_n$:
$$ R(T) = R(A_1, \dots, A_n)$$
*Definizione* (**schema di base di dati**): insieme di schemi di relazione:
$$ R = \{R_1(X_1), \dots, R_k(X_k)\}$$
Dove $X_1, \dots, X_k$ sono insiemi di attributi.

Una tupla su un insieme di attributi $T$, denotata con $t$, è una funzione che associa a ciascun attributo $A$ in $T$ un valore del dominio di $A$.
- $t[A]$ denota il valore della tupla $t$ sull'attributo $A \in T$
- $t[X]$ denota i valori della tupla $t$ sugli attributi $X \in T$

![[Screenshot 2023-11-22 at 18.27.10.png]]

*Definizione*: Un **istanza di relazione** o **relazione** su uno schema $R(X)$ è l'insieme $r$ di tuple su $X$.

*Definizione*: Un **istanza di base di dati** su uno schema $$R = \{R_1(X_1), \dots, R_n(X_n)\}$$
è l'insieme delle relazioni $r= \{r_1, \dots, r_n\}$ (con $r_i$ relazione su $R_i$

**Esempio**.
![[Screenshot 2023-11-22 at 18.30.14.png]]

I modello relazione impone ai dati una struttura rigida: le informazioni sono rappresentate per mezzo di ennuple, solo alcuni formati di ennuple sono ammessi: quelli che corrispondono agli schemi di relazione.

*Definizione* (**valore nullo**): denota l'assenza di un valore del dominio (e non è un valore del dominio).  $t[A]$ per ogni attributo $A$, è un valore del dominio $dom(A)$ oppure valore nullo NULL. 

![[Screenshot 2023-11-22 at 18.47.10.png | 400]]

# References