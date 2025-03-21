**Data time:** 12:58 - 26-11-2024

**Status**: #note #youngling 

**Tags:** [[Statistice riassuntive e grafici]] [[Statistics]]

**Area**: 
# Indici statistici

Dato un vettore di quantità numeriche $x = (x_1, ..., n_n) \in \mathbb{R}^n$ di dati gli indici riassumono proprietà significative dei dati.
###### Media campionaria (o media aritmetica)
$$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_{i}$$
La media campionaria si può anche calcolare partendo dalla **frequenza relativa** $p(a_j, x) = \frac{\# \{i : x_i = a_j\}}{n}$ che indica quanto un singolo dato compare nell'insieme. Abbiamo quindi che dati $a_1, ..., a_M$ possibili esiti:
$$\bar{x} = \sum_{i=1}^M a_{j}p(a_{j, x})$$
###### Mediana
È il valore all'interno dell'insieme tale per cui la metà dei valori è minore e l'altra metà è maggiore.
###### Varianza campionaria
$$var(x) = \frac{1}{n-1}\sum_{i=1}^n(x_{i}-\bar{x})^2$$
La varianza campionaria è una misura della dispersione calcolata su un campione di dati, non sull'intera popolazione. Si usa n-1 al denominatore per correggere la tendenza del campione a sottostimare la varianza della popolazione (**correzione di Bessel**)
###### Varianza empirica
$$var_{e}(x) = \frac{1}{n}\sum_{i=1}^n(x_{i}-\bar{x})^2$$
La varianza empirica è una misura della dispersione calcolata su un set di dati osservati. Non necessariamente si riferisce a un campione, ma più in generale a dati "empirici".
###### Scarto quadratico medio o deviazione standard
$$\sigma(x) = \sqrt{ var(x) }$$
È una misura statistica che indica quanto i dati di un insieme si discostano in media dalla loro media aritmetica. Il valore minimo è 0

![Deviazione standard e scarto quadratico medio: formule - WeSchool | 200](https://static.oilproject.org/content/6300/normale.jpg)

###### Misura campionaria di asimmetria
$$b = \frac{1}{\sigma^3} \cdot \frac{1}{n} \sum_{i=1}^n(x_{i} - \bar{x})^3$$
È un indice che misura l'asimmetria di una distribuzione attorno al suo valore medio.
- $n > 0$ indica asimmetria positiva
- $n < 0$ indica asimmetria negativa
# References