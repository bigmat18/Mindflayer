**Data time:** 01:27 - 05-12-2024

**Status**: #note #youngling 

**Tags:** [[Test statistici]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Z-Test

Si chiama z-test il **test sulla media [[Variabili Aleatorie Notevoli|gaussiana]] con varianza nota**. Consideriamo come partenza che l'ipotesi nulla sia data da un singolo valore per la media. Dobbiamo testare che la media $m$ coincida con un certo valore $m = m_0$. In altre parole abbiamo che:
$$\mathcal{H}_0) \:\:m = m_0 \:\:\:\:\:\:\:\mathcal{H}_1) \:\:m\neq m_0$$
#### Formulazione del test
Il test si formula con la regione critica
$$C = \bigg\{ \sqrt{n}\frac{|\overline{X}_n - m_0|}{\sigma}> q_{1 - \alpha/2}\bigg\} = \bigg\{  |\bar{X} - m_0| > \frac{\sigma}{\sqrt{n}}q_{1-\frac{\alpha}{2}}\bigg\}$$
- $\overline{X}$ media campionaria
- $m_0$ media attesa della popolazione sotto ipotesi nulla
- $\sigma$ deviazione standard
- $n$ dimensione campione

Per la spiegazione completa di come si è arrivati a questo risultato guardare le dispense. Notiamo che, analogamente al caso degli [[Intervalli di Fiducia|intervalli di fiducia]], l'ampiezza della regione di accettazione $C^c$ cioè $2\sigma q_{1-\alpha/2}/\sqrt{n}$ 
- **cresce** al crescere del livello del test $1 - \alpha$.
- **cresce** al crescere di $\sigma^2$.
- **decresce** al crescere di n.

#### Calcolo p-value
Partendo dalla definizione del [[Test statistici|p-value]] come probabilità rispetto all'ipotesi $\mathcal{H}_0$ andiamo ad ottenere
$$\bar{\alpha} = \bar{\alpha}(x_1, \dots, x_n) = \mathbb{P}_{m_0}\bigg(\sqrt{n}\frac{|\bar{X} - m_0|}{\sigma} > \frac{\sqrt{n}}{\sigma}|\bar{x}_n - m_0|\bigg) = 2 \bigg[1 - \Phi\bigg(\frac{\sqrt{n}}{\sigma}|\bar{x}_n - m_0|\bigg)\bigg]$$
# References