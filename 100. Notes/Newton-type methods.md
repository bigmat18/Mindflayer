---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# Metodi di tipo Newton (Newton-Type Methods)

Se vogliamo trovare una direzione di discesa migliore che porti a una convergenza più rapida, dobbiamo usare un modello migliore della funzione. Finora ci siamo basati su un modello lineare (il gradiente). Il passo logico successivo è passare a un **modello quadratico**.

## Il Metodo di Newton: Il caso (localmente) strettamente convesso

Quando la funzione è strettamente convessa nel punto corrente, la sua matrice Hessiana (la matrice delle derivate seconde) è definita positiva, ovvero $\nabla^2 f(x^i) > 0$. Questo garantisce l'esistenza di un minimo unico per la nostra approssimazione di Taylor del secondo ordine $Q_{x^i}(z)$.

Per trovare questo minimo, usiamo la **direzione di Newton**:

$$d^i = -[\nabla^2 f(x^i)]^{-1} \nabla f(x^i)$$

_Spiegazione:_ Invece di fare un passo semplicemente nella direzione di massima discesa (il gradiente negativo), moltiplichiamo il gradiente per l'inversa della matrice Hessiana. L'Hessiana cattura la curvatura della funzione. Facendo questo, stiamo effettivamente trovando il minimo esatto dell'approssimazione quadratica $Q_{x^i}(z)$ in un solo passo.

Poiché questa direzione punta esattamente al minimo del nostro modello, non c'è bisogno di calcolare la lunghezza del passo (stepsize): facciamo semplicemente il passo completo $\alpha^i = 1$.

La regola di aggiornamento per il metodo di Newton puro è semplicemente:

$$x^{i+1} = x^i + d^i$$

Un altro modo per interpretare questo metodo è vederlo come la risoluzione di un'**equazione non lineare**. Vogliamo trovare il punto in cui il gradiente è zero ($\nabla f(x) = 0$). Possiamo approssimare il gradiente usando un'espansione di Taylor del primo ordine:

$$\nabla f(x) \approx \nabla f(x^i) + \nabla^2 f(x^i)(x - x^i)$$

Ponendo questa espressione uguale a zero e risolvendo per $x$, otteniamo esattamente il passo di Newton.

Sappiamo anche che questa è garantita essere una direzione di discesa. Poiché $\nabla^2 f(x^i) > 0$, anche la sua inversa è definita positiva ($[\nabla^2 f(x^i)]^{-1} > 0$). Pertanto, la derivata direzionale è strettamente negativa:

$$\langle\nabla f(x^i), d^i\rangle = -\nabla f(x^i)^T [\nabla^2 f(x^i)]^{-1} \nabla f(x^i) < 0$$

_(Nota: sebbene sia negativa, dobbiamo comunque assicurarci che sia "abbastanza negativa" per garantire la convergenza)_.

## Convergenza (Globale) del Metodo di Newton

Il metodo di Newton puro è estremamente veloce, ma non è globalmente convergente: se si parte troppo lontani dal minimo, potrebbe divergere o oscillare. Per rimediare, creiamo il **Globalised Newton's method** aggiungendo semplicemente una Armijo-Wolfe Line Search (AWLS) o una Backtracking Line Search (BLS), ma testando sempre per primo il passo ideale $\alpha^0 = 1$.

Ci sono tre teoremi principali riguardo alla sua convergenza:

**Teorema 1 (Convergenza Globale):** Se $f \in C^2$ è L-smooth e $\tau$-convessa, il metodo globalizzato converge globalmente (tramite il teorema di Zoutendijk). L'angolo di discesa è limitato:

$$\cos(\theta^i) \le -\tau/L [< 0]$$

_Spiegazione:_ Poiché gli autovalori dell'Hessiana sono limitati tra $\tau$ e $L$, la direzione di Newton non potrà mai diventare perfettamente perpendicolare al gradiente.

**Teorema 2 (Convergenza Quadratica):** Se $f \in C^3$, all'ottimo $\nabla f(x_*) = 0$, e l'Hessiana è definita positiva $\nabla^2 f(x_*) > 0 \Rightarrow \exists \delta > 0$ tale che se partiamo abbastanza vicini all'ottimo ($x^0 \in \mathcal{B}(x_*, \delta)$), il "Newton puro" ($\alpha^i = 1$) convergerà a $x_*$ **quadraticamente**.

_Spiegazione:_ Convergenza quadratica significa che il numero di cifre decimali corrette all'incirca raddoppia a ogni singola iterazione. È una velocità sbalorditiva.

**Teorema 3 (La Transizione):** Se la sequenza $\{x^i\} \to x_*$, allora esisterà un'iterazione $h$ tale che il passo completo di Newton $\alpha^i = 1$ soddisfa perfettamente la condizione di Armijo (A) per tutti gli $i \ge h$. Questo richiede che il parametro di Armijo sia $m_1 < 1/2$ (un $m_1$ più grande rifiuterebbe artificialmente il vero minimo di una funzione quadratica).

_Spiegazione:_ L'algoritmo presenta naturalmente due fasi. Una "Fase Globale" in cui il passo $\alpha^i$ varia per navigare lo spazio in sicurezza, seguita automaticamente da una "Fase di Newton pura" in cui $\alpha^i = 1$ viene sempre accettato, innescando la convergenza quadratica. Questa fase pura di solito conclude l'ottimizzazione in $O(1)$ ($\approx 6$) iterazioni nella pratica.

### Matematicamente parlando: Bozza della Dimostrazione del Teorema 3

Per capire perché il passo completo $\alpha^i = 1$ viene alla fine accettato, usiamo l'espansione di Taylor di $f(x^i + d^i)$:

$$f(x^i + d^i) = f(x^i) + \langle\nabla f(x^i), d^i\rangle + \frac{1}{2}(d^i)^T [\nabla^2 f(x^i)] d^i + R(d^i)$$

Poiché $d^i$ è la direzione di Newton, sappiamo che $\nabla^2 f(x^i) d^i = -\nabla f(x^i)$. Sostituendo questo nel termine quadratico otteniamo:

$$= f(x^i) - \nabla f(x^i)^T [\nabla^2 f(x^i)]^{-1} \nabla f(x^i) + \frac{1}{2}\nabla f(x^i)^T [\nabla^2 f(x^i)]^{-1} \nabla f(x^i) + R(d^i)$$

Che si semplifica in modo elegante in:

$$= f(x^i) + \frac{1}{2}\langle\nabla f(x^i), d^i\rangle + R(d^i)$$

Man mano che ci avviciniamo al minimo, $d^i \to 0$. La derivata direzionale $\varphi_{x^i, d^i}'(0) = \langle\nabla f(x^i), d^i\rangle \to 0$, ma il resto di Taylor $R(d^i)$ tende a 0 ancora più velocemente. Alla fine, il resto è trascurabile, e il passo produce esattamente una frazione di $1/2$ della discesa promessa. Questo è il motivo per cui la condizione di Armijo è soddisfatta a patto che $m_1 < 1/2$.

---

## Interpretazione Geometrica: Newton = Gradiente + Dilatazione dello Spazio

C'è un'interpretazione geometrica incredibilmente elegante del metodo di Newton: è semplicemente il Metodo del Gradiente standard che opera in uno **spazio distorto (dilatato)**.

Consideriamo una funzione quadratica $f(x) = \frac{1}{2}x^T Q x + qx$, dove il passo di Newton è $d = -x - Q^{-1}q$. Facendo questo passo completo si ottiene $\nabla f(x + d) = 0$, il che significa che il metodo di Newton termina in esattamente un'iterazione.

Poiché $Q > 0$ (definita positiva), possiamo decomporla in $Q = R^T R$ (dove $R$ è una matrice non singolare).

Se applichiamo un cambio di variabili biettivo per "distorcere" il nostro spazio, definendo $z = Rx \equiv x = R^{-1}z$, la nostra funzione diventa:

$$h(z) = f(R^{-1}z) = \frac{1}{2}z^T I z + q R^{-1}z$$

In questo nuovo "spazio-z", l'Hessiana è semplicemente la matrice Identità $I$. Le curve di livello ellittiche della funzione sono state stirate fino a diventare cerchi perfetti!

In questo spazio perfettamente sferico, il gradiente standard $g = -\nabla h(z) = -z - R^{-1}q$ punta _esattamente_ al centro. Fare un passo di gradiente standard $\nabla h(z + g) = 0$ risolve il problema all'istante.

Se traduciamo questo gradiente magico $g$ dallo spazio-z al nostro spazio-x originale, otteniamo esattamente la direzione di Newton:

$$R^{-1}g = R^{-1}(-z - R^{-1}q) = -x - Q^{-1}q = d$$

---

## Il Caso Non Convesso e le Modifiche dell'Hessiana

Cosa succede se ci troviamo in una regione non convessa in cui l'Hessiana non è definita positiva? Se $\nabla^2 f(x^i)$ ha autovalori negativi (es., in un punto di sella), la direzione di Newton potrebbe puntare in salita, verso un massimo!

Per risolvere il problema, definiamo una direzione modificata $d^i \leftarrow -[H^i]^{-1} \nabla f(x^i)$, dove forziamo la matrice $H^i$ a essere definita positiva: $\tau I \le H^i \le LI$.

Se $\nabla^2 f \ne 0$, possiamo scegliere un $\epsilon^i$ "piccolo" tale da spostare la matrice:

$$H^i = \nabla^2 f(x^i) + \epsilon^i I > 0$$

_Spiegazione:_ Aggiungendo un multiplo della matrice Identità, stiamo aggiungendo $\epsilon^i$ a ogni autovalore. Se scegliamo $\epsilon^i$ in modo che sia leggermente più grande del valore assoluto dell'autovalore più negativo ($\lambda^n < 0$), tutti gli autovalori diventano positivi.

Una formula semplice per questo shift è $\epsilon = \max\{0, \delta - \lambda^n\}$ per un piccolo parametro $\delta$ scelto appropriatamente (come 1e-8 o 1e-12). Questo approccio risolve perfettamente il problema di ottimizzazione $\min\{||H - \nabla^2 f(x^i)||_2 : H \ge \delta I\}$.

Man mano che l'algoritmo si avvicina a un minimo locale stretto dove $\nabla^2 f(x_*) \ge \delta I$, lo shift $\epsilon^i$ diventa naturalmente $0$, il che significa $H^i = \nabla^2 f(x^i)$. L'algoritmo fa una transizione fluida tornando al metodo di Newton puro, riguadagnando la convergenza quadratica nella coda.

### Il Collo di Bottiglia Computazionale

Che si usi l'Hessiana esatta o una $H^i$ modificata, bisogna comunque risolvere un sistema lineare o calcolare una fattorizzazione di matrice (come la Cholesky $H^i = L^i (L^i)^T$). Questo richiede $O(n^3)$ operazioni. Per problemi su larga scala (es., $n = 10^4+$), $O(n^3)$ è semplicemente troppo costoso. Abbiamo bisogno di qualcosa di molto più economico, $O(n^2)$ o meno, il che ci porta verso l'approccio **Trust Region** e ai **Metodi Quasi-Newton**.

---

# Un Approccio Diverso: Trust Region (Regione di Fiducia)

I metodi visti finora (Line Search) seguono un approccio in due fasi: prima scelgono una direzione $d^i \in \mathbb{R}^n$, e poi cercano un passo $\alpha^i \in \mathbb{R}$ adeguato lungo quella direzione. I metodi **Trust Region (TR)** capovolgono completamente questa logica: prima si sceglie la lunghezza massima del passo (il "raggio di fiducia" $\alpha^i$ o $r$), e solo dopo si cerca la direzione ottimale all'interno di quel raggio.

## Il Problema della Curvatura Negativa

Nel metodo di Newton puro, se $\nabla^2 f(x^i)$ ha autovalori negativi, esistono direzioni di curvatura negativa lungo le quali $f$ decresce. Questi sono esattamente i punti in cui vogliamo andare per minimizzare $f$, quindi perché escluderli modificando l'Hessiana?

Il modello quadratico $Q^i(z)$ non ha un minimo globale su tutto $\mathbb{R}^n$ se ci sono curvature negative. Tuttavia, se vincoliamo la ricerca a un insieme compatto (una "regione di fiducia" $\mathcal{T}^i$ attorno a $x^i$ dove sappiamo che il nostro modello quadratico approssima bene la funzione reale), il minimo esiste sempre.

$$x^{i+1} \in \text{argmin} \{Q^i(z) : z \in \mathcal{T}^i\}$$

_Spiegazione:_ Stiamo risolvendo un problema di ottimizzazione vincolata a ogni iterazione. Se scegliamo come regione $\mathcal{T}^i$ una sfera euclidea $\mathcal{B}_2(x^i, r)$, il problema è risolvibile in modo efficiente ("round balls are simpler than kinky balls").

## La Soluzione Matematica del Trust Region

Sostituendo l'Hessiana esatta con una sua approssimazione $H^i \approx \nabla^2 f(x^i)$ (non necessariamente definita positiva), il punto ottimale $x^{i+1} = x^i + d^i$ per il sottomodello quadratico vincolato esiste ed è caratterizzato dalle seguenti condizioni (con $\exists \lambda^i \ge 0$):

1. $$H^i + \lambda^i I \ge 0$$
    
2. $$||d^i|| \le r$$
    
3. $$[H^i + \lambda^i I]d^i = -\nabla f(x^i)$$
    
4. $$\lambda^i(r - ||d^i||) = 0$$
    

_Spiegazione:_

- L'equazione 1 ci dice che l'aggiunta dello scalare $\lambda^i$ alla diagonale della matrice "corregge" l'Hessiana forzandola a diventare semidefinita positiva, risolvendo elegantemente il problema delle curvature negative.
    
- L'equazione 3 è una versione modificata del passo di Newton.
    
- L'equazione 4 è la condizione di complementarità: se il passo calcolato cade strettamente all'interno del raggio ($||d^i|| < r$), allora $\lambda^i = 0$ e stiamo facendo un puro passo di Newton standard (il vincolo non ha effetto). Man mano che la sequenza converge all'ottimo ($\{x^i\} \to x_*$), il passo diventa piccolo ($||d^i|| \to 0$), $\lambda^i = 0$ definitivamente e si riottiene la convergenza quadratica nella coda.
    

---

# Metodi Quasi-Newton

Calcolare e invertire (o fattorizzare) la matrice Hessiana esatta $\nabla^2 f(x^i)$ costa $O(n^3)$ operazioni, il che è impraticabile per problemi su larga scala. I metodi Quasi-Newton risolvono questo problema costruendo iterativamente una matrice $H^i$ che approssima l'Hessiana usando solo le informazioni raccolte dal gradiente nei passi precedenti ("learning $\nabla^2 f$ out of samples of $\nabla f$").

Lo spazio delle matrici $H^i$ che offrono una convergenza veloce ("superlineare") è grande. Si ottiene convergenza superlineare se $H^i$ si comporta come l'Hessiana vera lungo la direzione del passo appena effettuato ($d^i$): non ci interessa che sia accurata altrove.

## L'Equazione Secante (Secant Equation)

Definiamo la differenza tra le posizioni e la differenza tra i gradienti di due passi consecutivi:

$$s^i = x^{i+1} - x^i = \alpha^i d^i$$

$$y^i = \nabla f(x^{i+1}) - \nabla f(x^i)$$

Vogliamo che il nostro nuovo modello quadratico $m^{i+1}(x)$ concordi con la derivata appena osservata. Imponendo la condizione $\nabla m^{i+1}(x^i) = \nabla f(x^i)$, otteniamo l'**equazione secante**:

$$(S) \quad H^{i+1} s^i = y^i$$

_Spiegazione:_ Questa equazione forza la nuova matrice $H^{i+1}$ a mappare esattamente il passo fisico $s^i$ nella variazione del gradiente $y^i$. Moltiplicando a sinistra per $(s^i)^T$, otteniamo la **Curvature Condition**:

$$(C) \quad \langle s^i, y^i \rangle = (s^i)^T H^{i+1} s^i > 0$$

_(Spesso scritta come $\rho^i = 1 / \langle s^i, y^i \rangle > 0$)_.

_Spiegazione:_ Affinché $H^{i+1}$ sia definita positiva, questo prodotto scalare deve essere strettamente maggiore di zero. Fortunatamente, se usiamo una Line Search che rispetta la condizione di Wolfe forte $(W)$, la condizione di curvatura $(C)$ può sempre essere soddisfatta.

## DFP (Davidon-Fletcher-Powell)

Per trovare $H^{i+1}$, cerchiamo la matrice che soddisfi $(S)$, sia definita positiva ($H \ge 0$), e sia "il più vicina possibile" (minimizzando la distanza di Frobenius $||H - H^i||_F$) alla matrice precedente $H^i$. La soluzione è la formula DFP:

$$(DFP) \quad H^{i+1} = (I - \rho^i y^i (s^i)^T) H^i (I - \rho^i s^i (y^i)^T) + \rho^i y^i (y^i)^T$$

_Spiegazione:_ Poiché a noi serve in realtà l'inversa $B^{i+1} = [H^{i+1}]^{-1}$ per calcolare il passo di Newton, possiamo applicare la formula di _Sherman-Morrison-Woodbury_ (SMW) per aggiornare direttamente la matrice inversa:

$$(DFP-1) \quad B^{i+1} = B^i + \rho^i s^i (s^i)^T - B^i y^i (y^i)^T B^i / (y^i)^T B^i y^i$$

Grazie a questa formula, l'aggiornamento richiede solo prodotti matrice-vettore, riducendo il costo computazionale a **$O(n^2)$** per iterazione, senza mai dover calcolare un'inversa vera e propria.

## BFGS (Broyden-Fletcher-Goldfarb-Shanno)

La formula DFP è abbastanza efficiente, ma possiamo fare di meglio. Scrivendo l'equazione secante per $B^{i+1}$ ($s^i = B^{i+1}y^i$) e minimizzando la distanza dell'inversa, dato che tutto è simmetrico (basta scambiare $B \leftrightarrow H$ e $s \leftrightarrow y$), otteniamo la formula BFGS:

$$(BFGS) \quad H^{i+1} = H^i + \rho^i y^i (y^i)^T - H^i s^i (s^i)^T H^i / (s^i)^T H^i s^i$$

$$(BFGS) \quad B^{i+1} = (I - \rho^i s^i (y^i)^T) B^i (I - \rho^i y^i (s^i)^T) + \rho^i s^i (s^i)^T$$

_Spiegazione:_ BFGS costruisce un eccellente compromesso tra il costo per iterazione (che rimane $O(n^2)$) e la velocità di convergenza.

## Limited-Memory BFGS (L-BFGS)

Per problemi veramente grandi (es. $n$ molto elevato), anche $O(n^2)$ memoria/tempo per salvare la matrice densa $B$ è decisamente troppo.

La soluzione è **L-BFGS** ("Limited-memory BFGS"): invece di memorizzare esplicitamente la matrice $B^i$, "srotoliamo" le ultime iterazioni mantenendo in memoria solo gli ultimi $k$ vettori di aggiornamento $s$ e $y$ (con $k \ll n$).

Definendo $V^i = I - \rho^i y^i (s^i)^T$, l'aggiornamento assume la forma:

$$B^{i+1} = (V^{i-k}V^{i-k+1}...V^i)^T B^{i-k} (V^{i-k}V^{i-k+1}...V^i) + \dots + \rho^i s^i (s^i)^T$$

_Spiegazione:_ Quando dobbiamo calcolare il passo $d^i = -B^i \nabla f(x^i)$, ricostruiamo il risultato iterativamente applicando i prodotti vettore-vettore salvati. Il costo crolla a **$O(kn)$** per iterazione. C'è un trade-off: all'aumentare di $k$ si converge come il metodo di Newton, al diminuire di $k$ la convergenza peggiora e si comporta come il metodo del gradiente.

---

# Metodi del Gradiente Deflesso e Gradiente Coniugato Non Lineare (NCG)

Se anche $O(kn)$ è troppo o vogliamo evitare completamente di usare matrici, possiamo usare l'alternativa più economica: _Deflecting_ (Deflettere). L'idea è generare la nuova direzione $d^i$ deflettendo il gradiente corrente tramite la direzione calcolata all'iterazione precedente:

$$d^i = -\nabla f(x^i) + \beta^i d^{i-1}$$

Questo approccio è intrinsecamente **$O(n)$** per iterazione. Se poniamo $d^0 = -\nabla f(x^0)$, la direzione attuale diventa di fatto un aggregato di "tutti i gradienti passati", che funge da "storia" della computazione (simile a ciò che fa $H^i$ in BFGS).

I metodi **Nonlinear Conjugate Gradient (NCG)** utilizzano formule specifiche per calcolare il parametro scalare $\beta^i$. Alcune storiche sono:

1. **Fletcher-Reeves (FR):** $\beta_{FR}^i = ||\nabla f(x^i)||^2 / ||\nabla f(x^{i-1})||^2$
    
2. **Polak-Ribière (PR):** $\beta_{PR}^i = \langle \nabla f(x^i) - \nabla f(x^{i-1}), \nabla f(x^i) \rangle / ||\nabla f(x^{i-1})||^2$
    
3. **Dai-Yuan (DY):** $\beta_{DY}^i = ||\nabla f(x^i)||^2 / \langle \nabla f(x^i) - \nabla f(x^{i-1}), d^{i-1} \rangle$
    
4. **Hestenes-Stiefel (HS):** $\beta_{HS}^i = \langle \nabla f(x^i) - \nabla f(x^{i-1}), \nabla f(x^i) \rangle / \langle \nabla f(x^i) - \nabla f(x^{i-1}), d^{i-1} \rangle$
    

_Convergenza ed Efficienza:_ Se la funzione è perfettamente quadratica e si usa una Line Search esatta, il CG converge esattamente in $n$ iterazioni (o meno, se gli autovalori sono ben raggruppati tramite precondizionamento). L'efficienza è che $n$ passi di CG equivalgono approssimativamente a 1 passo di Newton.

Tuttavia, per funzioni non lineari generali, la convergenza dipende molto dalle formule e dalle condizioni. Se $||\nabla f(x^i)|| \ll ||d^i||$, l'algoritmo va in stallo e "un passo falso porta a molti passi falsi". Per questo motivo, di tanto in tanto si fa un **restart**: si ignora la storia e si azzera la deflessione ponendo semplicemente $d^i = -\nabla f(x^i)$ per riprendere il controllo. Nel complesso è un approccio potente, ma non facile da gestire.

---

# Metodi Heavy Ball Gradient (Momentum)

Un "gradiente coniugato per poveri" è il metodo dell'**Heavy Ball** (conosciuto in altri contesti come "Momentum"). Questo metodo utilizza un processo di aggiornamento leggermente diverso:

$$x^{i+1} \leftarrow x^i - \alpha^i \nabla f(x^i) + \beta^i (x^i - x^{i-1})$$

_Spiegazione:_ In questo schema, $x^i$ si comporta come un punto "pesante" che mantiene la direzione in cui stava già andando tramite il "momentum term" $\beta^i (x^i - x^{i-1})$, mentre la "forza" del gradiente $-\nabla f(x^i)$ devia e guida la traiettoria verso l'ottimo $x_*$.

Un momento (inerzia) $\beta^i$ elevato riduce gli "zig-zag" classici del gradiente, favorendo una convergenza migliore. L'algoritmo non è un metodo di pura discesa per $f$ (cioè non è garantito che $f(x^{i+1}) < f(x^i)$ ad ogni passo), ma con scelte appropriate dei parametri $\alpha$ e $\beta$ costanti si comporta come un algoritmo di discesa lineare rispetto alla distanza dall'ottimo $d^i$.

## Analisi Matematica (Tasso Ottimale)

Impostare l'Heavy Ball per funzionare richiede l'analisi di una complessa ricorrenza a due termini che unisce l'errore al passo $i$ e l'errore al passo $i-1$ in una matrice a blocchi. Tramite il Teorema del Valore Medio e la stima del raggio spettrale $\rho(C^i)$ della matrice risultante, si ricavano i valori ottimali.

Se il problema è L-smooth e $\tau$-convesso, scegliendo:

$$\alpha = \frac{4}{(\sqrt{L} + \sqrt{\tau})^2} \quad \text{e} \quad \sqrt{\beta} = \frac{\sqrt{L} - \sqrt{\tau}}{\sqrt{L} + \sqrt{\tau}} < 1$$

Il metodo dell'Heavy Ball ottiene un **tasso ottimale** di convergenza lineare:

$$r = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$$

_(dove $\kappa = L/\tau$)_.

_Confronto pratico:_ Se il problema è mal condizionato (es. $\kappa = 1000$), il metodo del gradiente classico con fixed stepsize avrebbe un tasso di convergenza estremamente lento $r \approx 0.996$. Con l'Heavy Ball, il tasso migliora drasticamente a $r \approx 0.938$. Può sembrare una differenza piccola, ma su 100 iterazioni il gradiente normale ridurrebbe l'errore moltiplicando per $0.996^{100} \approx 0.6698$, mentre l'Heavy ball lo ridurrebbe per $0.938^{100} \approx 0.0016$. Un miglioramento sostanziale in pratica per un metodo che costa appena $O(n)$ operazioni!
# References