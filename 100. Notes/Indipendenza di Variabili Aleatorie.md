**Data time:** 11:35 - 02-12-2024

**Status**: #note #youngling 

**Tags:** [[Variabili Aleatorie Multivalore]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Indipendenza di Variabili Aleatorie
Andiamo a codificare come ogni informazioni contenuta in una V.A. X sia indipende da quelle contenute in una V.A. Y. Dati X, Y sono **indipendenti** se presi comunque A,B sottoinsiemi di $\mathbb{R}$ gli eventi $X^{-1}(A), Y^{-1}(B)$ sono indipendenti, cioè vale
$$\mathbb{P}\{X \in A, Y \in B\} = \mathbb{P}\{X \in A\} \cdot \mathbb{P}\{Y \in B\}$$
#### Indipendenza V.A. discrete
Date due variabili aleatorie discrete X,Y con rispettivamente immagine nei punti $x_i, y_i$ queste sono indipendenti se e solo se vale l'eguaglianza tra le funzioni di massa
$$p(x_i, y_j) = p_X(x_i) \cdot p_Y(y_j) \:\:\forall(x_i, y_j)$$
#### Indipendenza V.A con densità
Nel caso di V.A. con densità invece, X e Y sono indipendenti se e solo se vale l'eguaglianza
$$f(x,y) = f_X(x) \cdot f_Y(y) \:\:\: \forall (x, y)$$
#### Indipendenza V.A. Binomiale
Se X e Y sono rispettivamente Binomiale $B(n,p), B(m, p)$ e sono indipendenza allora $Z = X + Y$ è binomiale $B(n+m, p)$

La somma di due V.A. è anche essa una V.A. che rappresenta tutti i possibili risultati della somma dei valori di X e Y. Per esempio se X e Y rappresentano il risultato del lancio di due danti la somma rappresenta i sultati da 2 a 12 e le loro probabilità

![[Screenshot 2024-12-02 at 12.29.33.png | 350]]
#### Indipendenza somma discrete
Siano X,Y variabili aleatorie discrete a valori naturali e indipendenti, e sia Z = X + Y e rispettivamente $p_X, p_Y, p_z$ funzioni di massa, si ha che
$$p_Z(n) = \sum_{h=0}^n p_X(h) \cdot p_Y(n-h)$$
#### Formula della convuluzione
Siano X e Y indipendenti con densità rispettivamente di $f_X, f_Y$ e sia Z = X + Y, la variabile Z ha densità
$$f_Z(z) = \int_{-\infty}^{+\infty}f_X(x)f_Y(z-x) dx = \int_{-\infty}^{+\infty}f_Y(y)f_X(z-y) dy$$
# References