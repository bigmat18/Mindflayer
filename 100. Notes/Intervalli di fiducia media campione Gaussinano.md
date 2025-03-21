**Data time:** 15:36 - 03-12-2024

**Status**: #note #youngling 

**Tags:** [[Intervalli di Fiducia]] [[Statistics]]

**Area**: 
# Intervalli di fiducia media campione Gaussinano

Sia $X_1, \dots, X_n$ un campione statistico con legge [[Variabili Aleatorie Notevoli|Gaussiana]] $N(m, \sigma^2)$. Tale legge ha in effetti due parametri, corrispondenti a media e varianza. In questo caso poniamo $\theta = m$ cioè alla media.

Essendo che $\bar{X}_n = (X_1, \dots, X_n) / n$ è uno stimatore per la media m del campione lo possiamo usare per punto ti partenza nell'intervallo di fiducia.

Se $[\bar{X}_n \pm d]$, con $d > 0$ una V.A. possibilmente costante, è un intervallo di fiducia per la media m, d detta **precisione della stima** e $d/\bar{X}_n$ è detta **precisione relativa della stima**.

La **precisione della stima** (il valore d) in un intervallo di fiducia si riferisce a quanto "stretto" o "ampio" è l'intervallo, cioè quanto siamo vicini alla vera stima del parametro di interesse. Una maggiore precisione implica un intervallo più stretto, mentre una precisione inferiore si traduce in un intervallo più ampio.

- Cresce al crescere del livello di fiducia $1 - \alpha$
- Cresce al decrescere di $\sigma^2$ 
- Decresce al crescere di n
#### Intervallo di fiducia per la media, varianza nota
Dato un $\alpha \in (0, 1)$ e assumendo sia nota $\sigma > 0$ ([[Valore Atteso Varianza e Momenti|varianza]]), l'intervallo aleatorio
$$\bigg[\bar{X}_n \pm \frac{\sigma}{\sqrt{n}}q_{1-\frac{\alpha}{2}}\bigg]$$
è un intervallo di fiducia per la media m del campione $X_1, \dots, X_n$ con livello di fiducia $1-\alpha$
#### Intervallo di fiducia per la media, varianza non nota
In questo caso non possiamo usare $\sigma^2$ essendo che è sconociuto ma possiamo usare un suo [[Campioni Statistici e Stimatori|stimatore]]
$$S_n^2 = \frac{\sum_{i=1}^n(X_i^2-\bar{X})}{n-1}$$
Usando questo, dato un $\alpha \in (0,1)$ l'intervallo aleatorio
$$\bigg[ \bar{X}_n \pm \frac{S_n}{\sqrt{n}} \tau_{1-\frac{\alpha}{2}, n-1} \bigg]$$
è un intervallo di fiducia per la media m del campione $X_1, \dots, X_n$ con livello di fiducia $1-\alpha$
#### Intervallo di fiducia unilaterale per la media, varianza nota
Gli intervalli appena descritti sono bilaterali perché entrambi gli estremi sono V.A. a volte può essere utile considerare un intervallo unilaterale, ad esempio se ci chiediamo che la media non sia più alta di un tot.

Dato quindi un $\alpha \in (0,1)$ gli intervalli aleatori
$$\bigg( -\infty, \bar{X}_n + \frac{\sigma}{\sqrt{n}}q_{1-\alpha}\bigg], \:\:\:\bigg[\bar{X}_n - \frac{\sigma}{\sqrt{n}} q_{1-\alpha}, +\infty \bigg)$$
sono intervalli di fiducia per la media m del campione $X_1, \dots, X_n$ con livello di fiducia $1-\alpha$ 
# References