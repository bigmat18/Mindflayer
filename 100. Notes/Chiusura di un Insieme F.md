**Data time:** 00:46 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Normalizzazione]]

**Area**: [[Bachelor's Degree]]
# Chiusura di un Insieme F

*Definizione* (**chiusura di F**): dato un insieme D di DF, la chiusura di F, denotata con $F^+$. è:
$$F^+ = \{X \rightarrow Y \:|\: F \: |- \: X \rightarrow Y\}$$
Un problema che si presenta spesso è quello di decidere se una dipendenza funzionale appartiene a $F^+$ (problema dell’implicazione); la sua risoluzione con l’algoritmo banale (di generare $F^+$ applicando ad F ripetutamente gli assiomi di Armstrong) ha una complessità esponenziale rispetto al numero di attributi dello schema.

*Definizione* (**Chiusura di X rispetto d F**): Dato R<T,F> e $X \subseteq T$ la chiusura di X rispetto ad F denota la $X_F^+$ (o $X^+$ se F è chiaro dal contesto)  è
$$X^+_F = \{A_i \in T \:|\: F|- \: X \rightarrow A_i\}$$

**Problemi dell'implicazione**: controllare se una DF V → W $\in$ $F^+$ Un algoritmo efficiente per risolvere il problema dell’implicazione senza calcolare la chiusura di F scaturisce dal seguente teorema.

*Teorema/Osservazione*: $F |- X \rightarrow Y \Leftrightarrow T \subseteq X_F^+$

**Algoritmo per calcolare $X^+_F$**
Sia X un insieme di attributi e F un insieme di dipendenze. Vogliamo calcolare $X^+_F$.
1. Inizializziamo $X^+$ con l'insieme X.
2. Se fra le dipendenze di F c'è una dipendenza $Y \rightarrow A$ con $Y \subseteq X^+$ allora si inserisce A in $X^+$ ossia $X^+ = X^+ \cup \{A\}$ 
3. Si ripete il passo 2 fino a quando non ci sono altri attributi da aggiungere ad $X^+$
4. Si da in output $X_F^+ = X^+$

**Chiusura Lenta**
input: R<T,F>  X $\subseteq$ T)
output: $X^+$
begin
	$X^+ = X$     Inizializziamo $X^+$ con l'insieme X
	while ($X^+$ cambia) do     Fino a quando non ci sono altri atrib da aggiungere a $X^+$ 
			for $W \to V$ in F with $W \subseteq X^+$ and $V \lor X^+$
				do $X^+ = X^+ \cup V$ 
			Se fra le dipendenze di F c’è una dipendenza W → V con W⊆ X+ allora si inserisce V in $X^+$ , ossia $X^+ = X^+ \cup \{V\}$ 
end

**L’algoritmo termina** perché ad ogni passo viene aggiunto un nuovo attributo a X+ . Essendo gli attributi in numero finito, a un certo punto l’algoritmo deve fermarsi.

Per dimostrare la correttezza, si dimostra che $X_F^+ + = X^+$ (per induzione)

**Esempio**
$F = \{DB \to E, B \to C, A \to B\}$ trovare $(AD)^+$:
Vogliamo conoscere gli attributi che sono determinati funzionalmente da un insieme di dipendenze A e D.
$X^+ = AD$,     $X^+ = ADB$,     $X^+ = ADBE$,      $X^+ = ADBEC$ 

Se fra le dipendenze di F c’è una dipendenza Y → A con $Y\subseteq X^+$ allora si inserisce A in $X^+$ , ossia $X^+ = X^+ \cup \{A\}$.

**[[Chiavi nel Modello Relazionale|Chiavi]] ed attributi primi**
*Definizione*: Dato lo schema R<T, F> diremo che un insieme di attributi $W \subseteq T$ è una **chiave candidata** di R se:
- $W \to T \in F^+$    (Si ricava tutto da W)
- $\forall V \subset W, V \to T \notin F^+$   (W è minimale)

**Attributo primo**: attributo che appartiene ad almeno una chiave.

**Esempio**. Prendiamo l'esempio di prima.
$F = \{DB \to E, B \to C, A \to B\}$ trovare $(AD)^+$:
Vogliamo conoscere gli attributi che sono determinati funzionalmente da un insieme di dipendenze A e D.
$X^+ = AD$,     $X^+ = ADB$,     $X^+ = ADBE$,      $X^+ = ADBEC$ 
- AD è superchiave? Si poiché contiene tutti gli attributi 
- A è superchiave? A→B, A→BC, si ferma → non è superchiave 
- ABD è superchiave? (ABD)+ è analoga a (AD)$^+$ , perché ABD è più grande di AD, quindi è superchiave
- ABC è superchiave? ABC stesso, quindi non è superchiave

Complessità: il problema di trovare tutte le chiavi di una relazione richiede un algoritmo di complessità esponenziale nel caso peggiore, il problema di controllare se un attributo è primo è NP- completo.

L’algoritmo per trovare tutte le chiavi si basa su due proprietà: 
1. Se un attributo A di T non appare a destra di alcuna dipendenza in F, allora A appartiene ad ogni chiave di R 
2. Se un attributo A di T appare a destra di qualche dipendenza in F, ma non appare a sinistra di alcuna dipendenza non banale, allora A non appartiene ad alcuna chiave.

# References