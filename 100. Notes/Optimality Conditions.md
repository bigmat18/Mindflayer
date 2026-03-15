---
Data: 
Tags:
  - note
  - youngling
Connection:
Area:
---
# Condizioni di Ottimalità

## Intuizione Grafica e il Caso 1D
Per comprendere l'ottimalità nel calcolo multivariato, ricordiamo prima come l'ottimalità locale e le derivate si relazionano graficamente in una semplice funzione 1D $f(x)$. 
- Se $f'(x) < 0$ oppure $f'(x) > 0$, il punto $x$ chiaramente non può essere un minimo locale perché muovendosi nella direzione della pendenza negativa si otterrà un valore della funzione più piccolo. 
- Quindi, $f'(x) = 0$ in tutti i minimi locali, e di conseguenza anche nel minimo globale. 
- Tuttavia, la condizione $f'(x) = 0$ non è esclusiva dei minimi; si verifica anche nei massimi locali (e globali), così come nei punti di sella. 



## Condizione di Ottimalità (Necessaria, Locale) del Primo Ordine
In $\mathbb{R}^n$, l'equivalente di $f'(x) = 0$ è il gradiente valutato come vettore nullo. La condizione necessaria formale stabilisce che se $f$ è differenziabile in $x$ e $x$ è un minimo locale, allora:
$$\nabla f(x) = 0$$
Un punto che soddisfa questa condizione è chiamato punto stazionario.

Poiché le dimostrazioni dei teoremi spesso generano gli algoritmi che usiamo per l'ottimizzazione, è importante analizzare la dimostrazione per assurdo. 
- Supponiamo che $x$ sia un minimo locale ma $\nabla f(x) \ne 0$.
- Dimostrare che $x$ non è un minimo locale non è immediato perché non possiamo controllare facilmente tutte le direzioni ($\not\equiv \forall/$). Dobbiamo dimostrare che $\forall\epsilon>0$ "abbastanza piccolo", $\exists z\in\mathcal{B}(x,\epsilon)$ tale che $f(z)<f(x)$.
- Questo significa che dobbiamo costruire infiniti punti $z$ migliori di $x$, arbitrariamente vicini ad esso.
- Fortunatamente, tutti questi punti $z$ possono essere presi lungo una singola direzione rettilinea $d\in\mathbb{R}^{n}$, definita come $z=x+\alpha d$ per un passo $\alpha>0$.
- Possiamo scegliere la direzione "migliore" $d$ (con $||d||=1$) in modo che $\frac{\partial f}{\partial d}(x)$ sia il più negativo possibile, il che corrisponde all'anti-gradiente normalizzato: $-\nabla f(x)(/||\nabla f(x)||)$.

### Matematicamente Parlando: La Dimostrazione del Primo Ordine
Valutiamo la tomografia $\varphi(\alpha)=\varphi_{x,-\nabla f(x)}(\alpha)$, non normalizzando intenzionalmente $d$ per mantenere la matematica più pulita. Vogliamo dimostrare che:
$$\exists\overline{\alpha}>0 \text{ t.c. } \varphi(\alpha)<f(x)=\varphi(0)\forall\alpha\in[0,\overline{\alpha}] \quad (1)$$

Usando il modello del primo ordine in $z$, il resto è $R(z-x)=f(z)-L_{x}(z)$. La definizione di $f\in C^{1}$ impone che il resto svanisca "più velocemente di $h\rightarrow0$":
$$\lim_{h\rightarrow0}R(h)/||h||=0 \equiv R(h)\rightarrow0$$

Sostituiamo $h = -\alpha\nabla f(x)$ nell'espansione di Taylor per ottenere la nostra equazione per $\varphi(\alpha)$:
$$\varphi(\alpha)=f(x-\alpha\nabla f(x))=f(x)+\langle-\alpha\nabla f(x),\nabla f(x)\rangle+R(-\alpha\nabla f(x))$$
$$\varphi(\alpha)=f(x)-\alpha||\nabla f(x)||^{2}+R(-\alpha\nabla f(x))$$
Questa equazione contrappone un termine negativo lineare in $\alpha$ a un resto (possibilmente) positivo "più che lineare".

Mentre $\alpha\rightarrow0$ (che implica $||h=-\alpha\nabla f(x)||\rightarrow0$), è chiaro chi vince la battaglia:
$$\lim_{\alpha\rightarrow0}R(-\alpha\nabla f(x))/||\alpha\nabla f(x)||=\lim_{h\rightarrow0}R(h)/||h||=0$$
Questa equivalenza significa che per qualsiasi limite di errore strettamente positivo, il resto è limitato da una frazione del termine lineare per un passo abbastanza piccolo:
$$\equiv\forall\epsilon>0\exists\overline{\alpha}>0 \text{ t.c. } R(-\alpha\nabla f(x))/\alpha||\nabla f(x)||\le\epsilon\forall\alpha\in[0,\overline{\alpha}]$$

Se prendiamo specificamente $\epsilon<||\nabla f(x)||$, otteniamo un limite superiore stretto sul resto:
$$R(-\alpha\nabla f(x))<\alpha||\nabla f(x)||^{2}$$
Sostituendo questo risultato, troviamo in modo conclusivo:
$$\varphi(\alpha)=f(x)-\alpha||\nabla f(x)||^{2}+R(-\alpha\nabla f(x))<f(x)$$
Questa dimostrazione mostra rigorosamente che fare un passo abbastanza piccolo lungo $-\nabla f(x)(\ne0)$ produce sempre un punto $z$ strettamente migliore.

## Condizioni di Ottimalità (Necessarie, Locali) del Secondo Ordine

Il modello del primo ordine è completamente "piatto" in un punto stazionario, il che significa che non può distinguere un minimo locale da un massimo o da un punto di sella. Per distinguerli, dobbiamo guardare la curvatura di $f$.

Se $f$ fosse puramente quadratica, guarderemmo gli autovalori della sua matrice $Q=\nabla^{2}f(x)$. L'idea ovvia è approssimare la funzione generale $f$ con una funzione quadratica, creando il modello del secondo ordine:
$$Q_{x}(z)=L_{x}(z)+\frac{1}{2}(z-x)^{T}\nabla^{2}f(x)(z-x)$$
In un punto stazionario, $\nabla Q_{x}(x)=\nabla L_{x}(x)=\nabla f(x)\Rightarrow\nabla Q_{x}(x)=0$. Di conseguenza, $\nabla^{2}f(x) \ge 0 \iff x$ è il minimo (globale) di $Q_{x}$.

Possiamo dimostrare che questo vale anche per $f$: se $f\in C^{2}$ e $x$ è un minimo locale, allora $\nabla^{2}f(x)\ge0$. Questo richiede il teorema di Taylor del secondo ordine, dove il resto $R(z-x)$ svanisce "più velocemente che quadraticamente":
$$f(z)=L_{x}(z)+\frac{1}{2}(z-x)^{T}\nabla^{2}f(x)(z-x)+R(z-x)$$
con $\lim_{h\rightarrow0}R(h)/||h||^{2}=0\equiv R(h)\rightarrow0$ più velocemente di $h^{2}\rightarrow0$.

### Matematicamente Parlando: La Dimostrazione del Secondo Ordine
Dimostriamo questo per assurdo. Assumiamo $f\in C^{2}$, $x$ è un minimo locale, ma l'Hessiana non è semi-definita positiva ($\nabla^{2}f(x)\not\ge0$). Questo significa che esiste una direzione di curvatura negativa $d$ (senza perdita di generalità, poniamo $||d||=1$) tale per cui:
$$d^{T}\nabla^{2}f(x)d<0$$

Valutando la tomografia $\varphi(\alpha)=\varphi_{x,d}(\alpha)$ usando l'espansione di Taylor del secondo ordine e sapendo che $\nabla f(x)=0\equiv L_{x}(z)=f(x)$, otteniamo:
$$\varphi(\alpha)=f(x)+\frac{1}{2}\alpha^{2}d^{T}\nabla^{2}f(x)d+R(\alpha d)$$
Questa equazione accoppia un termine quadratico negativo in $\alpha$ contro un resto "più che quadratico" (possibilmente) positivo.

Mentre $\alpha\rightarrow0$ (che equivale a $||h=\alpha d||$ poiché $||d||=1$), è chiaro chi vince:
$$\lim_{\alpha\rightarrow0}R(\alpha d)/\alpha^{2}=\lim_{h\rightarrow0}R(h)/||h||^{2}=0$$
$$\equiv\forall\epsilon>0\exists\overline{\alpha}>0 \text{ t.c. } R(\alpha d)\le\epsilon\alpha^{2}\forall\alpha\in[0,\overline{\alpha}]$$

Se scegliamo attentamente $(0<)\epsilon<-\frac{1}{2}d^{T}\nabla^{2}f(x)d$, limitiamo il resto in modo che $R(\alpha d)<-\frac{1}{2}\alpha^{2}d^{T}\nabla^{2}f(x)d$. Sostituendo questo otteniamo:
$$\varphi(\alpha) = f(x) + \frac{1}{2}\alpha^2d^T\nabla^2f(x)d + R(\alpha d) < f(x) \forall \alpha \in [0,\overline{\alpha}]$$
Quindi, in un minimo locale, non possono assolutamente esserci direzioni di curvatura negativa. La regola generale è: "quando la prima derivata è 0, prevalgono gli effetti del secondo ordine".

## Condizioni di Ottimalità (Sufficienti, Locali) del Secondo Ordine

La condizione necessaria è quasi sufficiente. Per $f\in C^{2}$, la condizione sufficiente stabilisce che:
$$\nabla f(x)=0 \text{ e } \nabla^{2}f(x)>0\Rightarrow x \text{ minimo locale}$$
Richiedere che l'Hessiana sia strettamente definita positiva ($>0$ anziché $\ge0$) evita il "caso peggiore" in cui $d^{T}\nabla^{2}f(x)d=0$. Una direzione a curvatura zero potrebbe nascondere un punto di sella (dove $f^{\prime\prime}(x)=0$), il che richiederebbe derivate di ordine ancora superiore per essere identificato correttamente.



Per dimostrare la sufficienza, usiamo l'espansione di Taylor del secondo ordine $f(x+d)=f(x)+\frac{1}{2}d^{T}\nabla^{2}f(x)d+R(d)$. Il limite del resto $\lim_{d\rightarrow0}R(d)/||d||^{2}=0$ implica:
$$\equiv\forall\epsilon>0\exists\delta>0 \text{ t.c. } R(d)/||d||^{2}\ge-\epsilon \equiv R(d)\ge-\epsilon||d||^{2}\forall d \text{ t.c. } ||d||<\delta$$

Sia $\lambda_{n}>0$ l'autovalore minimo di $\nabla^{2}f(x)$. Questo garantisce che $d^{T}\nabla^{2}f(x)d\ge\lambda_{n}||d||^{2}$ per qualsiasi direzione. 
Se prendiamo $\epsilon<\lambda_{n}/2$, allora per tutti i $d$ tali che $||d||<\delta$, otteniamo:
$$f(x+d)=f(x)+\frac{1}{2}d^{T}\nabla^{2}f(x)d+R(d)\ge f(x)+\frac{\lambda_{n}-\epsilon}{2}||d||^{2}$$

Poiché il termine $\frac{\lambda_{n}-\epsilon}{2}$ è strettamente positivo, prova molto più di quanto richiesto. Prova che $f$ cresce "almeno quadraticamente intorno a $x$":
$$\exists\delta>0 \text{ e } \gamma>0 \text{ t.c. } f(z)\ge f(x)+\gamma||z-x||^{2}\forall z\in\mathcal{B}(x,\delta)$$
Questa condizione è definita matematicamente come **ottimalità (locale) forte**.
# References