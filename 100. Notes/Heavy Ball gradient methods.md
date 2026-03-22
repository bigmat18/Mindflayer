---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# “Poorman’s conjugate gradient”: Heavy Ball Gradient

I metodi del gradiente coniugato e Quasi-Newton possono essere complessi e costosi. Esiste un'alternativa più semplice, nota come **Heavy Ball Gradient** (spesso chiamata semplicemente "Momentum" nel Machine Learning).

Il processo di aggiornamento è leggermente diverso dal solito:

$$x^{i+1} \leftarrow x^i - \alpha^i \nabla f(x^i) + \beta^i (x^i - x^{i-1})$$

_Spiegazione:_ * Il termine $\beta^i (x^i - x^{i-1})$ è il **"momentum term"** (termine di quantità di moto). L'idea fisica è che il punto $x^i$ sia "pesante" (heavy) e tenda a continuare a muoversi nella stessa direzione in cui si stava già muovendo.

- Contemporaneamente, la "forza" del gradiente $-\nabla f(x^i)$ sterza la traiettoria spingendola verso il minimo $x_*$.
    
- Un "momentum" $\beta^i$ grande significa meno "zig-zag" e una traiettoria più fluida.
    

A differenza del gradiente standard, è difficile garantire che $f(x^{i+1}) < f(x^i)$ ad ogni passo: di fatto, **non è un algoritmo di discesa per $f$**. Tuttavia, scegliendo appropriatamente $\alpha^i$ e $\beta^i$ costanti, si comporta come un algoritmo di **discesa lineare per la distanza $d$** dall'ottimo:

$$d^{i+1} = ||x^{i+1} - x_*|| \approx \le r ||x^i - x_*|| = d^i \quad \text{con} \quad r = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$$

Questo è il tasso ottimale raggiungibile. Per capirne l'impatto: se il numero di condizionamento è $\kappa = 1000$, il gradiente normale ha $r \approx 0.996$, mentre l'Heavy Ball ha $r \approx 0.938$. Dopo 100 iterazioni, l'errore del gradiente scende solo a $0.996^{100} = 0.6698$, mentre l'Heavy Ball lo abbatte a $0.938^{100} = 0.0016$. Una differenza abissale nella pratica!

---

## Analisi Matematica I: La Ricorrenza a Due Termini

Per dimostrare rigorosamente questo tasso di convergenza, si parte dalla definizione della ricorrenza, che nell'Heavy Ball dipende da due termini precedenti ($x^i$ e $x^{i-1}$). Possiamo scriverla in forma matriciale a blocchi:

$$\begin{bmatrix} x^{i+1} - x_* \\ x^i - x_* \end{bmatrix} = \begin{bmatrix} x^i + \beta^i(x^i - x^{i-1}) - \alpha^i(\nabla f(x^i) - \nabla f(x_*)) - x_* \\ x^i - x_* \end{bmatrix}$$

(Nota: abbiamo sottratto e aggiunto $\nabla f(x)$ che vale $0$ all'ottimo)

Applicando il Teorema del Valore Medio al gradiente, sappiamo che esiste un punto $w^i \in [x_*, x^i]$ tale che $\nabla f(x^i) - \nabla f(x_*) = \nabla^2 f(w^i)(x^i - x_*)$. Sostituendo questo termine:

$$= \begin{bmatrix} (x^i - x_*) - \alpha^i \nabla^2 f(w^i)(x^i - x_*) + \beta^i(x^i - x^{i-1}) \\ x^i - x_* \end{bmatrix}$$

Raggruppando i termini per isolare $(x^i - x_*)$ e aggiungendo/sottraendo $\beta^i x_*$:

$$= \begin{bmatrix} [I - \alpha^i \nabla^2 f(w^i)](x^i - x_*) + \beta^i(x^i - x^{i-1}) + \beta^i x_* - \beta^i x_* \\ x^i - x_* \end{bmatrix}$$

$$= \begin{bmatrix} [I - \alpha^i \nabla^2 f(w^i) + \beta^i I](x^i - x_*) - \beta^i(x^{i-1} - x_*) \\ x^i - x_* \end{bmatrix}$$

Infine, estraiamo la matrice di iterazione $C^i$:

$$= \begin{bmatrix} (1 + \beta^i)I - \alpha^i \nabla^2 f(w^i) & -\beta^i I \\ I & 0 \end{bmatrix} \begin{bmatrix} x^i - x_* \\ x^{i-1} - x_* \end{bmatrix}$$

Se potessimo trovare $\alpha^i$ e $\beta^i$ tali che la norma di questa matrice $||C^i|| < 1$, avremmo dimostrato la convergenza lineare. Purtroppo, non è così semplice, perché $||C^i|| > 1$.

---

## Analisi Matematica II: Raggio Spettrale (Complicato)

Poiché $C^i$ non è simmetrica, la sua norma è maggiore o uguale al suo **raggio spettrale** $\rho(C^i) = \max_j \{|\lambda_j(C^i)|\}$ (dove gli autovalori possono essere complessi, quindi $|\cdot|$ è il modulo, non il valore assoluto normale).

Attraverso una complessa diagonalizzazione a blocchi, il raggio spettrale $\rho(C^i)$ si scompone nel massimo dei raggi spettrali di $n$ matricine $2 \times 2$:

$$C_j = \begin{bmatrix} 1 + \beta^i - \alpha^i \lambda_j(D) & -\beta^i \\ 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

_(dove $\lambda_j(D)$ sono gli autovalori dell'Hessiana)_.

Risolvendo il polinomio caratteristico di queste matricine (un processo estremamente tedioso), si ottiene un limite superiore per il raggio spettrale:

$$\rho(C^i) \le \sqrt{\beta^i} = \max\{|1 - \sqrt{\alpha^i \tau}|, |1 - \sqrt{\alpha^i L}|\}$$

Per minimizzare questo massimo, l'**$\alpha$ ottimale** risulta essere:

$$\alpha = \frac{4}{(\sqrt{L} + \sqrt{\tau})^2} \implies \sqrt{\beta} = \frac{\sqrt{L} - \sqrt{\tau}}{\sqrt{L} + \sqrt{\tau}} < 1$$

Questo ci restituisce esattamente il tasso di convergenza ottimale $r = \sqrt{\beta} = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$. Questo varrebbe se potessimo dimostrare la convergenza lineare direttamente con $r = \sqrt{\beta}$, il che è _quasi_ vero, ma non del tutto.

---

## Analisi Matematica III: Formula di Gelfand (++Complicato)

Facciamo un'assunzione semplificatrice: supponiamo che $f$ sia quadratica. Questo rende l'Hessiana $\nabla^2 f$ costante, e di conseguenza la matrice di iterazione $C^i = C$ è costante.

Per ricorsione, l'errore al passo $i$ è limitato da:

$$||E^i|| \le ||C^i|| \cdot ||E^0||$$

_(dove $C^i$ è la matrice elevata alla $i$-esima potenza)_.

Qui entra in gioco la **Formula di Gelfand**:

$$\rho(C) = \lim_{i \to \infty} ||C^i||^{1/i}$$

Questo implica matematicamente che:

$$\forall \epsilon > 0 \quad \exists h \quad \text{s.t.} \quad ||C^i|| \le (\rho(C) + \epsilon)^i \quad \forall i \ge h$$

_Cosa significa in pratica?_ Significa che l'errore può crescere o oscillare all'inizio, ma, se aspettiamo un numero sufficiente di iterazioni $h$ ("large"), prima o poi l'algoritmo **"inizia a convergere"** con un tasso quasi lineare dettato dal raggio spettrale $\rho(C)$.

Nel caso in cui la funzione $f$ sia **non convessa**, l'algoritmo converge comunque se $\beta \in [0, 1)$ e $\alpha \in (0, 2(1-\beta)/L)$, sebbene la finestra per scegliere $\alpha$ diventi molto stretta all'avvicinarsi di $\beta \to 1$.

---

## E se $\tau = 0$? (Accelerated Gradient Method)

Cosa succede se la funzione non è fortemente convessa ($\tau = 0$)?

L'Heavy Ball in questo caso garantisce solo un tasso di errore $O(1/i)$, che teoricamente non è migliore del gradiente standard.

Per superare questo limite teorico, si utilizza una variante chiamata **Accelerated Gradient** (spesso noto come Metodo di Nesterov), il cui pseudocodice nasconde della vera e propria "magia nera" matematica:

Plaintext

```
procedure y = ACG(f, x, \epsilon)
    x_- <- x; \gamma <- 1;
    do { // warning: black magic ahead
        \gamma_- <- \gamma; 
        \gamma <- (\sqrt{4\gamma_-^2 + \gamma_-^4} - \gamma_-^2)/2; 
        \beta <- \gamma(1/\gamma_- - 1);
        y <- x + \beta(x - x_-); 
        g <- \nabla f(y); 
        x_- <- x; 
        x <- y - (1/L)g;
    } while( ||g|| > \epsilon );
```

_Spiegazione delle differenze:_ L'ACG è molto simile all'Heavy Ball, ma con una differenza cruciale: **il gradiente viene calcolato sul punto "previsto" dal momentum ($y$), e non sul punto corrente ($x$)**.

Questo piccolo cambiamento teorico assicura il tasso di convergenza **ottimale** possibile per funzioni solamente L-smooth: $O(LD^2/\sqrt{\epsilon})$. Se la funzione è anche $\tau$-convessa, ottiene lo stesso tasso lineare ottimale dell'Heavy Ball.

Tuttavia, nella pratica l'ACG è costantemente un po' lento ("slowish"): è stato accuratamente progettato per garantire matematicamente il miglior comportamento possibile nel _caso peggiore_ (worst-case behaviour), e si comporta esattamente come è stato programmato per fare.
# References