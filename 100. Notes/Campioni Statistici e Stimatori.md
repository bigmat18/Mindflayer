**Data time:** 12:23 - 03-12-2024

**Status**: #note #youngling 

**Tags:** [[Campioni di Variabili Aleatorie]] [[Statistics]]

**Area**: 
# Campioni Statistici e Stimatori

Partiamo assumendo di avere uno spazio di probabilità ed una variabile aleatoria X la cui distribuzione $\mathbb{P}_X$ è secondo i casi parzialmente o del tutto sconosciuta. Per determinarla abbiamo bisogno di $X_1, \dots, X_n$ variabili aleatorie indipendenti e aventi la stessa legge di $X$ 

Data $F = F_X$ c.d.f ([[Funzione di Ripartizione (CDF) e Quantili (PPF)|funzione di ripartizione]]) di una V.A. X, una famiglia di V.A. $X_1, \dots, X_n$ iid ([[Indipendenza di Variabili Aleatorie|indipendenti]] e equidistribuite) con legge data dalla cdf F si dice **campione statistico** o **campione aletorio** della V.A. X di numerosità n.

Nei casi la probabilità $\mathbb{P}_X$ partendo da $X_1, \dots, X_n$ non è possibile con questi strumenti dire molto, ma a volte la distribuzione $\mathbb{P}_X$ è parzialmente specificata ma mancano dei parametri che vengono indicati con $\theta_1, \dots, \theta_n$.

Per esempio nel caso di una gaussiana non sono specificati ne la media campionaria ne la varianza. A questo punto l'obbiettivo della stima parametrica è **ricostruire i parametri a partire dalle osservazioni**, cioè in funzione del campione.

Una funzione $g(X_1, \dots, X_n)$ di un campione statistico è chiamata **statistica campionaria**, oppure è detta anche **stimatore**. Alcuni esempi:
- **Media campionaria** $\bar{X} = \frac{X_1 + \dots + X_n}{n}$
- **Varianza campionaria** $S^2 = \frac{\sum_{i=1}^n(X_i - \bar{X})^2}{n-1}$

Dato un parametro $\theta$ della distribuzione ed un campione $X_1, \dots, X_n$ una statistica $g(X_1, \dots, X_n)$ si dice uno **stimatore corretto** del parametro $\theta$ se ammette [[Valore Atteso Varianza e Momenti|momento primo]] e
$$\mathbb{E}[g(X_1, \dots, X_n)] = \theta$$
cioè la media dell'estimatore è il parametro $\theta$.
Questa definizione garantisce che la media campionaria e la varianza campionaria siano stimatori corretti (guarda il libro per vedere la dimostrazione).

Dato un parametro $\theta$ della distribuzione e un campione $X_1, \dots, X_n$ di infinite iid di X, la successione di statistiche $g_n(X_1, \dots, X_n)$ si dice uno stimatore **consistente** di $\theta$ se, per $n \to \infty$, $g_n(X_1, \dots, X_n)$ tende a $\theta$ in probabilità, cioè se, per ogni $\epsilon > 0$
$$\lim_{n\to\infty}\mathbb{P}\{|g_n(X_1, \dots, X_m) - \theta| > \epsilon\} = 0$$

In altre parole, quando la taglia del campione diventa molto grande $g(X_1, \dots, X_n)$ si avvicina con alta probabilità al parametro $\theta$

Dato un parametro $\theta$ della distribuzione ed un campione $X_1, \dots, X_{n_0}$ con $n_0 \geq n, m$, dati due stimatori $g(X_1, \dots, X_n)$ e $h(X_1, \dots, X_n)$ che ammettono momento secondo, diciamo che $g(X_1, \dots X_m)$ è **più efficiente** di $h(X_1, \dots, X_n)$ se
$$Var(g(X_1, \dots, X_n)) \leq Var(h(X_1, \dots, X_n))$$
# References