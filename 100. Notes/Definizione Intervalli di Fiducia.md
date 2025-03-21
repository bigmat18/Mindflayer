**Data time:** 14:37 - 03-12-2024

**Status**: #note #youngling 

**Tags:** [[Intervalli di Fiducia]] [[Statistics]]

**Area**: 
# Definizione Intervalli di Fiducia

Consideriamo un campione statistico $X_1, \dots, X_n$ la cui legge dipende da un parametro $\theta \in Theta \subseteq \mathbb{R}$ 
$$\mathbb{P}_\theta(X_1 \leq t) = \dots = \mathbb{P}_{\theta}(X_n \leq t) = F_{\theta}(t)$$
In altre parole assumiamo che nello spazio di probabilità esista una famiglia di probabilità tali che per ogni $\theta$ fissato le variabili $X_1, \dots, X_n$ siano [[Indipendenza di Variabili Aleatorie|iid]] e con [[Funzione di Ripartizione (CDF) e Quantili (PPF)|cfd]] $F_\theta$.

L'obbiettivo rimane quello di trovare il più preciso valore di $\theta$ partendo dagli esiti del campione. 
Un intervallo di fiducia nello specifico è un intervallo i cui estremi sono calcolati a partire dai valori assunti da $X_1, \dots,X_n$, dunque un intervallo casuale nel quale ci aspettiamo che sia contenuto il parametro $\theta$ 

Dato un campione statistico $X_1, \dots, X_n$ di legge $\mathbb{P}$ con $\theta \in \Theta \subseteq \mathbb{R}$ e un numero $\alpha \in (0,1)$ definiamo un **intervallo aleatorio** come
$$I = [a(X_1, \dots, X_n), b(X_1, \dots, X_n)]$$
Dove $a, b: \mathbb{R}^n \to \mathbb{R}$ sono funzioni misurabili e quindi sono a loro volta V.A. Da qui si dice **intervallo di fiducia** per $\theta$ al livello $(1-\alpha)$ se
$$\forall \theta \in \Theta \:\:\:\: \mathbb{P}(\theta \in I) = \mathbb{P}_{\theta}(a(X_1, \dots, X_n) \leq \theta, b(X_1, \dots, X_n) \geq \theta) \geq 1- \alpha$$

Tipicamente $\alpha$ è un numero piccolo il modo che il livello di fiducia $1-\alpha$ si avvicini a 1. Però non è sempre possibile prendere dei valori di $\alpha$ troppo piccoli perché si rischia che $\theta$ non cada nell'intervallo.

Dato un determinato livello di fiducia per esempio al 95% vuol dire che se ripetessimo l'esperimento molte vole, nel 95% dei casi l'intervallo calcolato conterebbe la vera media della popolazione.

![[Screenshot 2024-12-03 at 15.35.59.png | 400]]

# References