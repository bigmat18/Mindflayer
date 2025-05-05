**Data time:** 01:58 - 05-12-2024

**Status**: #note #youngling 

**Tags:** [[Test statistici]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Z-Test unilaterale

Occupiamoci in questo caso di uno [[Z-Test]] per la media campionaria [[Variabili Aleatorie Notevoli|Gaussiana]] con ipotesi alternative.
$$\mathcal{H}_0) \:\:m \leq m_0 \:\:\:\:\:\:\:\mathcal{H}_1) \:\:m > m_0$$
#### Formulazione del test
L'intuizione ci spinge a rifiutare l'ipotesi se $(\bar{X} - m_0)$ è troppo grande, cioè a considerare una regione critica della forma $C = \{(\bar{X} - M_0) > d\}$ e la condizione sul livello diventa
$$\mathbb{P}_m((\bar{X} - M_0) > d) \leq \alpha$$
La probabilità sopra scritta cresce al crescere di arrivando a
$$C = \bigg\{  |\bar{X} - m_0| > \frac{\sigma}{\sqrt{n}}q_{1-\alpha}\bigg\}$$
#### Calcolo p-value
In questo caso i dati $(y_1, \dots, y_n)$ sono più estremi di $(x_1, \dots, x_n)$ rispetto ad $\mathcal{H}_0: m < m_0$ se $\bar{y}_n - m_0 > \bar{x}_n - m_0$ abbiamo quindi che il p-value, partendo dalla probabilità di avere dati "più estremi" è
$$\bar{\alpha} = \bar{\alpha}(x_1, \dots, x_n) = \mathbb{P}_{m_0}\bigg(\sqrt{n}\frac{|\bar{X} - m_0|}{\sigma} > \frac{\sqrt{n}}{\sigma}|\bar{x}_n - m_0|\bigg) = 1 - \Phi\bigg(\frac{\sqrt{n}}{\sigma}|\bar{x}_n - m_0|\bigg)$$
# References