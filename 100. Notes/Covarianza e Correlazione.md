**Data time:** 12:32 - 02-12-2024

**Status**: #note #youngling 

**Tags:** [[Variabili Aleatorie Multivalore]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Covarianza e Correlazione

Supponiamo di avare X e Y V.A. con valore atteso, allora anche X + Y ha valore atteso e valgono le seguenti proprietà:
- $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
- se $X \geq Y$ allora $\mathbb{E}[X] \geq \mathbb{E}[Y]$

Mentre le somma ha sempre valore atteso, nel caso del prodotto non è detto che ci sia. L'unico caso che garantisce che il prodotto fra due V.A. abbia valore atteso è se X e Y sono indipendenti, in questo caso
$$\mathbb{E}[X\cdot Y] = \mathbb{E}[X] \cdot \mathbb{E}[Y]$$
Inoltre, sempre se X e Y sono indipendenti e date due $h, f: \mathbb{R} \to \mathbb{R}$ vale che
$$\mathbb{E}[h(X)k(Y)] = \mathbb{E}[h(X)]\cdot\mathbb{E}[k(Y)]$$
#### Disuguaglianza di Schwartz
Siano X e Y due V.A. vale la seguente disuguaglianza
$$\mathbb{E}[|XY|] \leq \sqrt{\mathbb{E}[X^2]} \cdot \sqrt{\mathbb{E}[Y^2]}$$
Da questa disuguaglianza possiamo dire che se X e Y hanno il momento secondo il prodotto XY ha valore atteso.

#### Covarianza e Coefficiente di correlazione
Si chiama **covarianza** tra X e Y il numero
$$Cov(X, Y) = \mathbb{E}[(X- \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$
Questo valore indica il grado di relazioni lineare tra due variabili aleatorie, in altre parole descrive come due variabili aleatorie si comportano insieme, se crescono o diminuiscono contemporaneamente o vanno in direzioni opposte.
- **Segno positivo**. Abbiamo che se X aumenta anche Y tende ad umentare
- **Segno negativo**. Se X aumenta Y tende a diminuire
- **Zero**. Non c'è relazione lineare tra X e Y. Potrebbero esserci relazioni non lineari si dicono **scorrelate**

Questo numero però non fornisce informazioni dirette sull'intensità della relazione, per fare ciò si usa il coefficiente di correlazione

Se $Var(X) \neq 0, Var(Y) \neq 0$ si chiama **coefficiente di correlazione** il numero:
$$\rho(X, Y) = \frac{Cov(X,Y)}{\sigma(X)\sigma(Y)}$$
Dove $\sigma$ ricordiamo che è la [[Valore Atteso Varianza e Momenti|deviazione standard]].

Abbiamo che $-1 \leq \rho \leq 1$ e quindi che
- $\rho = 1$ correlazione lineare positiva perfetta (Y aumenta proporzionalmente a X)
- $\rho = -1$ correlazione lineare negativa perfetta (T diminuisce proporzionalmente a X)
- $\rho = 0$ Nessuna correlazione lineare (le variabili possono essere indipendenti o avere una relazione non lineare)

![[Screenshot 2024-12-02 at 13.12.14.png]]
# References