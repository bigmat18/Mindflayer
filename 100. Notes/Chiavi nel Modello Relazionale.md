**Data time:** 16:02 - 10-05-2025

**Status**: #note #youngling 

**Tags:** [[Basi di Dati]] [[Modello Relazionale]]

**Area**: [[Bachelor's Degree]]
# Chiavi nel Modello Relazionale

Informalmente una chiave è un insieme di attributi che identificano le ennuple di una relazione. Mentre formalmente:

*Definizione*: Un insieme $K$ di attributi è superchiave per $r$ se non contiene due ennuple (distinte) $t_1$ e $t_2$ con $t_1[K] = t_2[K]$. 

$K$ è una chiave per $r$ se è una **superchiave minimale** per $r$ (cioè non contiene un'altra superchiave).

**Esempio**
![[Screenshot 2023-11-22 at 19.04.41.png]]
Matricola è una chiave visto che è una superchiave ed è minimale. Mentre per esempio cognome, nome, nascita non può essere una superchiave minimale perché non è sempre vero.

I vincoli corrispondono a proprietà del mondo reale modellato dalla base di dati, interessano a livello di schema (con riferimento cioè a tutte le istanze possibili), ad uno schema associamo un insieme di vincoli e consideriamo corrette (valide, ammissibili) le istanze che soddisfano tutti i vincoli, un'istanza può soddisfare altri vincoli (“per caso”).

Una relazione non può contenere ennuple distinte ma con valori uguali (una relazione è un sottoinsieme del prodotto cartesiano). Ogni relazione ha sicuramente come superchiave **l’insieme di tutti gli attributi su cui è definita** e quindi ogni relazione ha (almeno) una chiave.

L’esistenza delle chiavi garantisce l’accessibilità a ciascun dato della base di dati, le chiavi permettono di correlare i dati in relazioni diverse: il modello relazionale è basato su valori.

**Chiavi e valori nulli**
La presenza di valori nulli fra i valori di una chiave non permette di identificare le ennuple di realizzare facilmente i riferimenti da altre relazioni.
![[Screenshot 2023-11-22 at 19.19.46.png]]

**Chiave primaria** 
Una chiave primaria è una chiave su cui non sono ammessi valori nulli. Notazione: sottolineatura.![[Screenshot 2023-11-22 at 19.20.27.png]]

# References