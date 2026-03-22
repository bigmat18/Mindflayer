---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# Metodi del Gradiente Coniugato Non Lineare (Nonlinear Conjugate Gradient Methods)

Come visto nella sezione precedente, l'idea del "deflecting" consiste nel calcolare la nuova direzione di discesa combinando il gradiente attuale con la direzione usata al passo precedente ($d^i = -\nabla f(x^i) + \beta^i d^{i-1}$).

I metodi del **Gradiente Coniugato Non Lineare (NCG)** rappresentano l'applicazione più sofisticata di questa idea: utilizzano formule matematiche specifiche per calcolare il parametro scalare $\beta^i$ in modo tale da simulare l'efficienza dei metodi di secondo ordine (come Newton) usando solo informazioni del primo ordine.

## L'Algoritmo NCG

La struttura generale dell'algoritmo NCG è molto simile a quella del normale metodo del gradiente, ma incorpora una "memoria" del passo precedente:

Plaintext

```
procedure x = NCG(f, x, \epsilon)
    \nabla f^- = 0;
    while( ||\nabla f(x)|| > \epsilon) do
        if (\nabla f^- == 0) then d <- -\nabla f(x);
        else { \beta = (right deflection value); d <- -\nabla f(x) + \beta d^- }
        \alpha <- LS(f,x,d); 
        x <- x + \alpha d; 
        d^- <- d; 
        \nabla f^- <- \nabla f(x);
```

_Spiegazione:_ * Al primo passo (o quando resettiamo l'algoritmo ponendo $\nabla f^- = 0$), partiamo semplicemente seguendo l'antigradiente puro: $d = -\nabla f(x)$.

- Nei passi successivi, calcoliamo un valore $\beta$ e lo usiamo per deflettere la traiettoria: $d \leftarrow -\nabla f(x) + \beta d^-$.
    
- Trovata la direzione, usiamo una Line Search (LS) per calcolare il passo $\alpha$, aggiorniamo la posizione $x$ e salviamo la direzione e il gradiente attuali per usarli al ciclo successivo.
    

## Le Formule per $\beta$

Esistono molteplici formule matematiche per calcolare il parametro di deflessione $\beta^i$. Le quattro più celebri e utilizzate storicamente sono:

1. **Fletcher-Reeves (FR):**
    
    $$\beta_{FR}^i = ||\nabla f(x^i)||^2 / ||\nabla f(x^{i-1})||^2$$
    
2. **Polak-Ribière (PR):**
    
    $$\beta_{PR}^i = \langle\nabla f(x^i) - \nabla f(x^{i-1}), \nabla f(x^i)\rangle / ||\nabla f(x^{i-1})||^2$$
    
3. **Hestenes-Stiefel (HS):**
    
    $$\beta_{HS}^i = \langle\nabla f(x^i) - \nabla f(x^{i-1}), \nabla f(x^i)\rangle / \langle\nabla f(x^i) - \nabla f(x^{i-1}), d^{i-1}\rangle$$
    
4. **Dai-Yuan (DY):**
    
    $$\beta_{DY}^i = ||\nabla f(x^i)||^2 / \langle\nabla f(x^i) - \nabla f(x^{i-1}), d^{i-1}\rangle$$
    

_Spiegazione (Perché così tante formule?):_

Se stiamo minimizzando una funzione _perfettamente quadratica_ ($f(x) = \frac{1}{2}x^T Q x + qx$) e utilizziamo una Line Search _esatta_, **tutte queste formule sono matematicamente equivalenti** e producono esattamente la stessa sequenza di passi. In quel caso ideale, l'algoritmo del Gradiente Coniugato converge in esattamente $n$ iterazioni (dove $n$ è la dimensione dello spazio, supponendo aritmetica esatta). Può metterci anche meno iterazioni se gli autovalori dell'Hessiana sono raggruppati in cluster (un effetto che si ottiene tramite il "precondizionamento").

Tuttavia, quando applichiamo questi metodi a **funzioni non lineari generiche**, e usiamo una Line Search inesatta (come la Armijo-Wolfe, AWLS), i percorsi generati da queste formule divergono drasticamente. Ad esempio, il numeratore di PR e HS usa la differenza $\nabla f(x^i) - \nabla f(x^{i-1})$, un termine che cattura implicitamente le informazioni sulla curvatura (Hessiana) della funzione non lineare, rendendole in pratica spesso superiori a FR.

---

## Convergenza e il Trucco del "Restart"

Dimostrare la convergenza globale per i metodi NCG è notoriamente complesso e dipende strettamente dalla formula $\beta$ scelta e dalle condizioni imposte:

- **Fletcher-Reeves (F-R):** Richiede una condizione di Wolfe molto forte sui parametri, ovvero $m_1 < m_2 < 1/2$, affinché l'intersezione $(A) \cap (W')$ funzioni a dovere.
    
- **Polak-Ribière (P-R):** Si può dimostrare matematicamente che la versione standard di P-R non converge su alcune funzioni. Per garantirne la convergenza, è necessario forzare $(A) \cap (W')$ e limitare $\beta$ inferiormente a zero con la variante **PR+**:
    
    $$\beta_{PR+}^i = \max\{\beta_{PR}^i, 0\}$$
    
    Un approccio simile, $\beta_{HS+}^i = \max\{\beta_{HS}^i, 0\}$, è utile e raccomandato per Hestenes-Stiefel.
    

**L'Importanza dei Restart:**

Imporre che $\beta$ non scenda sotto zero (come in PR+) è di fatto un **restart** (riavvio) dell'algoritmo: se la formula restituisce un valore negativo, lo tronchiamo a 0, il che significa ignorare la memoria passata e calcolare un puro passo di gradiente $d^i = -\nabla f(x^i)$.

I restart regolari sono un'idea eccellente, specialmente per Fletcher-Reeves. Il motivo è geometrico:

$$||\nabla f(x^i)|| \ll ||d^i|| \iff \cos(\theta^i) \approx 0 \equiv \nabla f(x^i) \approx \perp d^i$$

_Spiegazione:_ Se il gradiente diventa quasi ortogonale (perpendicolare) alla direzione di discesa, il passo $\alpha$ risulterà microscopico ($x^{i+1} \approx x^i$), il che porterà a un altro angolo pessimo al passo successivo ($\cos(\theta^{i+1}) \approx 0$). In altre parole: **"un passo sbagliato porta a molti passi sbagliati" (one bad step leads to many bad steps)**.

Azzerare la memoria fa uscire l'algoritmo da questo vicolo cieco. In effetti, i restart aiutano enormemente nelle dimostrazioni teoriche di convergenza, perché garantiscono che asintoticamente la deflessione svanisca e il gradiente puro faccia il grosso del lavoro. Spesso si imposta un restart automatico ogni $n$ passi, anche se questo approccio non è molto elegante quando $n$ è molto grande o molto piccolo.

---

## Efficienza Pratica e Teorica

L'efficienza dei metodi del gradiente coniugato non lineare può essere vista come una proprietà di "convergenza quadratica a $n$-passi":

$$||x^{i+n} - x_*|| \le r ||x^i - x_*||^2$$

_Spiegazione:_ La formula ci dice che $n$ passi di un metodo CG equivalgono approssimativamente a **1 singolo passo di un metodo di Newton**. Questo ha senso geometricamente: quando siamo "vicini a $X_*$", la funzione è molto ben approssimata dalla sua espansione di Taylor di secondo ordine ($f(\cdot) \approx Q_{x_*}(\cdot)$). E, come abbiamo visto, "in $n$ passi il CG risolve esattamente una funzione quadratica".

Sebbene l'idea di dover aspettare $n$ iterazioni non sia eccezionale quando lavoriamo con milioni di variabili, il NCG possiede interessanti relazioni con i metodi Quasi-Newton, sfociando in algoritmi ibridi potenti.

Nella pratica, le prestazioni dei vari $\beta$ variano sorprendentemente: Polak-Ribière o Dai-Yuan sono spesso le varianti migliori, ma i risultati possono variare enormemente a seconda del problema.

**In conclusione:** L'approccio NCG è potente ed estremamente efficiente in termini di memoria ($O(n)$), ma le sue idiosincrasie matematiche e la dipendenza dalle tolleranze della Line Search lo rendono "non facile da gestire" (not easy to manage) rispetto a un solido L-BFGS.

# References