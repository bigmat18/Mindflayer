**Data time:** 00:46 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Normalizzazione]]

**Area**: [[Bachelor's Degree]]
# Copertura Canonica

*Definizione*: Due insiemi di DF, F e G sullo schema R sono **equivalenti**.
$$F \equiv G \Leftrightarrow F^+ = G^+$$
Se $F \equiv G$ allora F è una **copertura** di G (e G una copertura di F).

*Definizione*: Sia F un insieme di DF. Dato una $X \to Y \in F$ si dice che X contiene un **attributo estraneo** $A_i$ sse $(X-\{A_i\}) \to Y \in F^+$ cioè $F |- (X - \{A_i\}) \to Y$

Come facciamo a stabilire che in una DF del tipo AX → B l’attributo A è estraneo? Per verificare se A è estraneo calcoliamo $X^+$ e verifichiamo se include B, ovvero se basta X a determinare B

**Esempio**. studenti(matricola, CF, Cognome, Nome, Anno)

se vale: 
- Docente, Giorno, Ora -> CodAula
- Docente, Giorno -> Ora
allora: Docente, Giorno -> CodAula, (quindi) nella prima dipendenza Ora è attributo estraneo.

*Definizione*: Sia D un insieme di DF. $X \to Y$ è una **dipendenza ridondante** sse $(F - \{X \to Y\})^+ = F^+$, Equivalentemente: $F - \{X \to Y\} |- X \to Y$.

Come facciamo a stabilire che una DF del tipo X → A è ridondante? La eliminiamo da F, calcoliamo $X^+$ e verifichiamo se include A, ovvero se con le DF che restano riusciamo ancora a dimostrare che X determina A.

**Esempio**. Orari(CodAula, NomeAula, Piano, Posti, Materia, CDL, Docente, Giorno, Ora)
se vale:
- Docente, Giorno, Ora -> CodAula 
- CodAula -> NomeAula
è inutile avere anche: Docente, Giorno, Ora -> NomeAula

**Esempio**. F = {B → C, B → A, C → A} 
B → A è ridondante poiché possiamo dedurla da B → C e C → A

*Definizione*. Sia F un insieme di DF. F è detta **copertura canonica** sse.
- la parte destra di ogni DF in F è attributo
- non esistono attributi estranei
- nessuna dipendenza in F è ridondante.

*Teorema*: Per ogni insieme di dipendenze F esiste una copertura canonica.

L'algoritmo per calcolare una copertura canonica è:
1. Trasformare le dipendenze nella forma $X \to A$. Si sostituisce l’insieme dato con quello equivalente che ha tutti i secondi membri costituiti da singoli attributi (dipendenze atomiche)
2. Eliminare gli attributi estranei. Per ogni dipendenza si verifica se esistono attributi eliminabili dal primo membro. Data una X → Y $\in$ F, si dice che X contiene un attributo estraneo $A_i$ sse $(X – \{A_i\}) \to Y \in F^+$ , cioè $F |- (X – \{A_i\}) \to Y$.
3. Eliminare le dipendenze ridondanti. $X \to Y$ è una dipendenza ridondante sse $(F - \{X \to Y\})^+ = F^+$, Equivalentemente $F - \{X \to Y\} |- X \to Y$

**Esempio**. Impiegato (Matricola, Cognome, Grado, Retribuzione, Dipartimento, Supervisore, Progetto, Anzianità).

Consideriamo il seguente insieme di dipendenze funzionali:
{M → RSDG, MS → CD, G → R, D →S, S → D, MPD → AM}
1. Creiamo le dipendenze funzionali atomiche: {M → R, M → S , M → D, M → G, MS → C, MS → D, G → R, D →S, S → D, MPD → A, MPD → M}
2. Eliminare gli attributi estranei:
	- è possibile eliminare S dal primo membro di MS → C e MS → D perché M → S (si ottiene da M → D e D → S)
	- È inoltre possibile eliminare D dal primo membro di MP<s>D</s> → A e MP<s>D</s> → M poiché M → D {M → R, M → S , M → D, M → G, M → C, M → D, G → R, D →S, S → D, MP → A, MP → M}
3. Si trova l’insieme di dipendenza funzionali non ridondante: eliminiamo le dipendenze ottenibili da altre:
	- M → R (deriva da M → G e G → R) 
	- M → S (deriva da M → D e D →S) 
	- M → D (Perché già M → D) 
	- MP → M (Perché M compare a primo membro)
# References