**Data time:** 14:02 - 28-11-2024

**Status**: #note #youngling 

**Tags:** [[Variabili Aleatore]] [[Statistics]]

**Area**: 
# Funzione di Ripartizione (CDF) e Quantili (PPF)

La funzione di ripartizione o (c.d.g) di una V.A. X è la funzione:
$$F_X: \mathbb{R} \to [0,1] \:\:\:\:\:F_X(x) = \mathbb{P}\{X \leq x\}$$
Questa funzione associa ad ogni valore reale x la probabilità che X assuma un valore minore o uguale a x.
- **intervallo di probabilità**: si calcola facendo $F_X(b) - F_X(a)$ in un intervallo $[a,b]$ 
$$\mathbb{P}\{a \leq X \leq b\} = F(b) - F(a)$$
- **punti di probabilità**: sono gli scalini della funzione.

![Funzione di ripartizione empirica - Wikipedia | 300](https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Empirical_distribution_function.png/800px-Empirical_distribution_function.png)

Alcune caratteristiche di una funzione di ripartizione sono:
- **F è non decrescente** ovvero se $x < y$ allora $F(x) \leq F(y)$
- Valgono i seguenti limiti
$$\lim_{x\to -\infty}F(x) = 0 \:\:\:\:\:\: \lim_{x \to + \infty} F(x) = 1$$
- **F è continua a destra** ossia per ogni $x \in \mathbb{R}$ vale $F(x_n) \to F(x)$ per ogni successione $x_n \to x$ con $x_n \geq x$ 

Una funzione di ripartizione è definita per qualsiasi V.A. e la sua struttura può variare nel caso di discrete e con densità. 
- Caso **V.A. discrete**
$$F_X(t) = \sum_{x_i \leq t} p(x_i)$$
- Caso **V.A. con densità**
$$F_X(t) = \int_{-\infty}^t f_X(t) dt$$
Nel caso di V.A. continue (con densità) è possibile ricavare f facendo la derivata di $F_X(t)$ 
$$f(t) = \frac{dF_X(t)}{dt}$$

##### Quantili (PPF)
Assegnata una V.A. ed un numero $\beta$ con $0 < \beta < 1$ si chiama $\beta$-quantile un numero reale r tale che si abbia $\mathbb{P}(X \leq r) \geq \beta$ e $\mathbb{P}(X \geq r) \geq 1 - \beta$ 

Il beta-quantile rappresenta una soglia sotto la quale si trova una proporzione $\beta$ della distribuzione di probabilità. Questa rappresentazione può essere vista anche l'inversa della CDF o **PPF**, probability point function.

![[Screenshot 2024-11-28 at 17.37.08.png | 400]]
# References