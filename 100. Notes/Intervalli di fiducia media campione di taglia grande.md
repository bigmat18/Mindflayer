**Data time:** 17:30 - 03-12-2024

**Status**: #note #youngling 

**Tags:** [[Intervalli di Fiducia]] [[Statistics]]

**Area**: 
# Intervalli di fiducia media campione di taglia grande

Lo stesso concetto utilizzato per gli intervalli di fiducia con [[Intervalli di fiducia media campione Bernoulli|campioni di bernoulli]] si può applicare esattamente nello stesso modo a campioni di variabili aleatorie di taglia grande 
#### Varianza nota
Sia $X_1, \dots, X_n$ un campione di V.A. iid, la cui legge ha momento secondo finito con n grande. Dato un $\alpha \in (0,1)$ l'intervallo aleatorio
$$\bigg[  \bar{X}_n \pm \frac{\sigma}{\sqrt{n}} q_{1-\frac{\alpha}{2}} \bigg]$$
è un intervallo di fiducia per la media m del campione con livello di fiducia **approssimativamente** $1-\alpha$. Precisamente si ha:
$$\lim_{n \to +\infty} \mathbb{P}\bigg(m \in \bigg[ \bar{X}_n \pm \frac{\sigma}{\sqrt{n}} q_{1-\frac{\alpha}{2}}\bigg]\bigg) = 1-\alpha$$
#### Varianza non nota
Sia $X_1, \dots, X_n$ un campione di V.A. iid, la cui legge ha momento secondo finito con n grande. Dato un $\alpha \in (0,1)$ l'intervallo aleatorio
$$\bigg[  \bar{X}_n \pm \frac{S_n}{\sqrt{n}} q_{1-\frac{\alpha}{2}} \bigg]$$
è un intervallo di fiducia per la media m del campione con livello di fiducia **approssimativamente** $1-\alpha$. Precisamente si ha:
$$\lim_{n \to +\infty} \mathbb{P}\bigg(m \in \bigg[ \bar{X}_n \pm \frac{S_n}{\sqrt{n}} q_{1-\frac{\alpha}{2}}\bigg]\bigg) = 1-\alpha$$
# References