**Data time:** 13:12 - 02-12-2024

**Status**: #note #youngling 

**Tags:** [[Variabili Aleatorie Multivalore]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Teorema Limite di Probabilità

Questo teorema si occupa di verificare il fenomeno per cui, dato una prova ripetuta con n alto, essa convergerà ad un risultato. Mentre il teorema centrale del limite indica quando può oscillare il numero di valori.

Consideriamo una famiglia di V.A. $X_1, \dots, X_n$ che siano **[[Indipendenza di Variabili Aleatorie|indipendenti]] e equidistribuite** (i.i.e). Equivalentemente sono iie se hanno tutte la stessa funzione di ripartizione
$$\mathbb{P}_{X_n}(t) = \mathbb{P}(X_n \leq t) = F(t)$$
e se vale
$$\mathbb{P}(X_{k_1} \leq \dots \leq X_{k_n} \leq t_n) = F_{X_1}(t_1) \cdots F_{X_n}(t_n)$$
Date V.A. X e $X_1, \dots, X_n$ sullo stesso spazio di probabilità si dice che $X_n$ **converge in probabilità** a X per $n \to +\infty$ se vale
$$\forall \epsilon > 0, \:\:\:\lim_{n \to +\infty} \mathbb{P}(|X_n - X| > \epsilon) = 0$$
Questo vuol dire che al crescere di n, la probabilità che $X_n$ sia "molto vicina" a X diventa arbitrariamente alta.

Il limite può anche essere una costante $X = c$. Da questa definizione possiamo dire che se
$$\lim_{n\to +\infty}\mathbb{E}[X_n] = c \in \mathbb{R}, \:\:\:\lim_{n\to +\infty}Var(X_n) = 0$$
allora la succession $(X_n)_{n> 1}$ converge in probabilità alla costante c.

#### Legge debole dei grandi numeri
Sia $X_1, \dots, X_n$ una successione i.i.e. dotate di momento secondo finito, e sia $\mu = \mathbb{E}[X_i]$ il loro valore atteso, allora la [[Indici statistici|media aritmetica]] $\bar{X}_n$ converge in probabilità a $\mu$ per $n\to +\infty$ per ogni $\epsilon > 0$ 
$$\lim_{n\to +\infty}\mathbb{P}\bigg(\bigg|\frac{X_1+ \dots+X_n}{n} - \mu\bigg|\bigg) = 0$$
Questo teorema dice che, più grande è il campione di osservazioni, più precisa sarà la stima della "vera media". Una conseguenza della legge dei grandi numeri è il comportamento, per n grandi, della seguente V.A.
$$S_n^2 := \frac{\sum_{i=1}^{n}(X_i - \bar{X}_n)^2}{n-1}$$
Diciamo che inoltre, data una successione di $X_1, \dots, X_n$ V.A. iid dotate di momento quarto, e sia $\sigma^2 = Var(X_i)$ la loro varianza. Per $n\to \infty$, $S_n$ converge in probabilità $\sigma$ ovvero
$$\forall \epsilon > 0 \:\:\: \lim_{n\to \infty} \mathbb{P}(|S_n^2 -\sigma^2| > \epsilon) = 0$$

Siano $(X_n)_{n \geq 1}$ una successione di v.a. ed X una variabile aleatoria, siano rispettivamente $F_n$ e $F$ le rispettive funzioni di ripartizioni supponiamo che F sia continua. Si dice che **la successione converge ad X** se per ogni t si ha:
$$\lim_{n\to\infty}F_n(t) = F(t)$$
#### Teorema centrale del limite
Sia $X_1, \dots, X_n$ una successione di variaibli iid di momento secondo finito, con valore atteso $E[X_i] = \mu$ e varianza $\sigma^2(X_i) = \sigma^2 > 0$. Presi due valori $-\infty \leq a \leq b \leq +\infty$ si ha
$$\lim_{n \to +\infty} \mathbb{P}\bigg(a \leq \frac{X_1 + \dots + X_n - n\mu}{\sigma\sqrt{n}} \leq b\bigg) = \frac{1}{\sqrt{2\pi}}\int_a^be^{-x^2/2}dx = \Phi(b) - \Phi(a)$$
Esso descrive il comportamento della somma di variabili aleatorie indipendenti, e afferma che, sotto certe condizioni, la distribuzione della somma normalizzata tende a una distribuzione normale, indipendentemente dalla distribuzione originale delle variabili aleatorie.

- **Sottrazione di $n\mu$** serve a centrare la somma attorno a 0
- **Divisione per $\sigma\sqrt{n}$** serve a standardizzare la somma, cioè rendere confrontabile con altre somme portandola su una scala che non dipende dalla dimensione del campione.

Agli effetti pratici possiamo dire che nell'ipotesi del teorema vale
$$\frac{X_i + \dots + X_n - n\mu}{\sigma \sqrt{n}} = \sqrt{n}\frac{\bar{X}_n - \mu}{\sigma}$$
Quando n è grande si approssima ad una Gaussiana standard. Spesso questo teorema si applica ad una sommatoria di [[Variabili Aleatorie Notevoli|variabili di bernolli]], in questo caso, essono dipendenti da n e p, si scrive
$$\frac{X - np}{\sqrt{np(1-p)}}$$
Che si approssima ad una gaussiana standard.
# References