**Data time:** 14:30 - 26-11-2024

**Status**: #note #youngling 

**Tags:** [[Dati Multivarianti]] [[Statistics]]

**Area**: 
# Covarianza e correlazione campionaria

Consideriamo un insieme di n coppie di dati numerici $(x, y) = (x_1, y_1), ...,, (x_n, y_n) \in \mathbb{R}^{2\times n}$ 
###### Covarianza campionaria
$$cov(v) = \sum_{i=1}^n\frac{(x_{i} - \bar{x})(y_{i} - \bar{y})}{n-1}$$
La covarianza misura come due variabili variano insieme:
- Se entrambi aumentano o diminuiscono contemporaneamente la covarianza è **positiva**
- Se una aumenta e l'altra diminuisce, la covarianza è **negativa**
- Se non c'è una relazione lineare, al covarianza sarà **vicina a zero**
###### Covarianza empirica
$$cov(v) = \sum_{i=1}^n\frac{(x_{i} - \bar{x})(y_{i} - \bar{y})}{n}$$
È la covarianza calcolata direttamente dai dati osservati (empirici), senza fare correzioni per la stima della popolazione.
###### Coefficiente di correlazione (di Pearson)
Supponiamo di avere che le [[Indici statistici|deviazioni standard]] $\sigma(x) \neq 0$ e $\sigma(y) \neq 0$. Si chiama **coefficiente di correlazione** il numero
$$r(x,y) = \frac{cov(x,y)}{\sigma(x)\sigma (y)}$$
Fra due variabili indica un'eventuale relazione di linearià fra esse. È un numero compreso tra -1 e +1. Indica quanto i cambiamenti in una variabili sono associati a cambiamenti nell'altra:
- **valori positivi** quando una variabile aumenta, l'altra tende ad aumentare
- **valori negativi** quando una variabile aumenta, l'altra tende a diminuire
- **valore pari a 0** indica assenza di correlazione lineare.

# References