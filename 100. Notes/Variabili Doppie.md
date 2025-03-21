**Data time:** 16:27 - 01-12-2024

**Status**: #note #youngling 

**Tags:** [[Variabili Aleatorie Multivalore]] [[Statistics]]

**Area**: 
# Variabili Doppie
Consideriamo due variabili aleatorie $X, Y: \Omega \to \mathbb{R}$ il valore $(X, Y)$ può essere visto come una funzione 
$$(X, Y): \Omega \to \mathbb{R}^2$$
Analogamente al caso delle V.A. singole definiamo la legge per le V.A. doppie
$$\mathbb{P}_{(X, Y)}(A) = \mathbb{P}((X, Y) \in A) = \mathbb{P}\{\omega \in \Omega: (X(\omega), Y(\omega)) \in A\}$$
Se X e Y rappresentano due caratteri di un dato esperimento la variabile doppia rappresenta la coppia di un dato dei due caratteri. Per esempio altezza e peso di un determinato campione.

Possiamo considerare le probabilità singole delle due V.A. $\mathbb{P}_X, \mathbb{P}_Y$ che sono dette **distribuzioni marginali**. Singolarmente queste distribuzioni non contengono tutte le informazioni di $\mathbb{P}_{(X,Y)}$ ma da esse si possono ricavare

#### Variabili doppie discrete
Una variabile aleatoria doppia $(X, Y)$ è detta **discreta** se la sua immagine è concentrato in un insieme finito o numerabile di (x, y), in questo caso, identificando la funzione di massa come $p(x_i, y_i) = \mathbb{P}(X = x_i, Y = y_i)$ si ha per $A \subseteq \mathbb{R}^2$ 
$$\mathbb{P}_{(X, Y)}(A) = \mathbb{P}\{(X, Y) \in A\} = \sum_{(x_i, y_j) \in A}p(x_i, y_j)$$
Dato una variabile aleatoria doppia (X,Y) discreta con funzione di massa $p(x_i, y_j)$ le sue componenti hanno funzioni di massa del tipo
$$p_X(x_i) = \sum_{y_j}p(x_i, y_j) \:\:\:\:p_Y(y_j) = \sum_{x_i}p(x_i, y_j)$$
#### Variabili doppie con densità
In caso di V.A. doppia (X, Y) con densità esiste funzione di massa integrabile e con $\int\int_{\mathbb{R}^2}f(x, y)dxdy = 1$ tale che valga
$$\mathbb{P}_{X,Y}(A) = \mathbb{P}\{(X,Y) \in A\} = \int\int_A f(x, y) dxdy$$
# References