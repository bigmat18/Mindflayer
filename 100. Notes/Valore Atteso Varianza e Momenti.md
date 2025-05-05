**Data time:** 15:22 - 01-12-2024

**Status**: #note #youngling 

**Tags:** [[Variabili Aleatore]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Valore Atteso Varianza e Momenti

### Valore attesa o Momento primo
Data una V.A. discreta con [[Probabilità sulla retta reale|funzioni di massa]] $p$ si dice che X ha valore atteso se $\sum_i |x_i| p(x_i) < +\infty$ in tal caso si chiama valore atteso il numero 
$$\mathbb{E}[X] = \sum_i x_ip(x_i)$$
Per una variabile aleatoria con densità abbiamo con le stesse pre-condizioni.
$$\mathbb{E}[X] = \int_{-\infty}^{+\infty} x_if(x_i)$$
Questo numero rappresenta la media ponderata di tutti i possibili valori che la variabile può assumere pesati secondo la loro probabilità.

Se prendiamo una V.A. X discreta la variabile $g(X)$ ammette valore atteso se $\sum_i |g(x)|p(x_i) < +\infty$ 
$$\mathbb{E}[g(X)] = \sum_i g(x_i)p(x_i)$$
Mentre se abbiamo che X è con densità abbiamo con le stesse pre-condizioni
$$\mathbb{E}[g(X)] = \int_{-\infty}^{+\infty} g(x_i)f(x_i)dx$$

Data una V.A X sia dicreta che con densità valgono le seguenti proprietà:
- per ogni $a, b \in \mathbb{R}$ $\mathbb{E}[aX + b] = a \mathbb{E}[X] + b$ in particolare $\mathbb{E}(b) = b$
- $|\mathbb{E}[X]| \leq \mathbb{E}[X]$
- se $\mathbb{P}(X \geq 0) = 1$ allora $\mathbb{E}[X] \geq 0$ 
##### Momento di ordine n
Possiamo estendere il concetto di momento di ordine prima a quello di momento di ordine n dicendo che se abbiamo una V.V X che $\mathbb{E}[|X|^n] < +\infty$ allora ammette momento di ordine n $\mathbb{E}[X^n]$

Questo è un concetto più generale del momento che permette di descrivere la varianza, la misura di asimmetrica ed altri fenomeni.
##### Disuguaglianza di Markov
Se una V.A. (discreta o continua) ha valore positivo $a > 0$ vale
$$a\mathbb{P}\{X \geq a\} \leq \mathbb{E}[X]$$
### Varianza
La varianza di una V.A. X si definisce come
$$Var(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$$
La **varianza** è una misura di dispersione che quantifica quanto i valori di una variabile aleatoria si distanziano, in media, dal loro valore atteso (media).

![[Screenshot 2024-12-01 at 16.17.34.png | 350]]

##### Scarto quadratico medio o deviazione standard
Si chiama scarto quadratico medio o deviazione standard di X la seguente formula
$$\sigma(X) \sqrt{Var(X)}$$
Questo valore serve per riportare la misura alla stessa unità della variabile originale e quantifica in quanto, in media, i valori di una distribuzione differiscono dalla media.

Per la varianza è valida la seguente equazione
$$Var(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$
### Disuguaglianza di Chebyshev
Se X è una variable aleatori e $d > 0$ vale che
$$\mathbb{P}\{|X - \mathbb{E}[X]| > d\} \leq \frac{Var(X)}{d^2}$$
Questo teorema viene utilizzato per misurare la dispersione dei dati per ogni distribuzione. Descrive qual è la percentuale dei dati che sarà entro un determinato numero di deviazioni standard della media, e questo funziona per qualsiasi curva di distribuzione di qualsiasi forma.
### Valori Attesi e varianze notevoli

# References