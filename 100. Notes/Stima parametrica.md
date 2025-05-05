**Data time:** 13:30 - 03-12-2024

**Status**: #note #youngling 

**Tags:** [[Campioni Statistici e Stimatori]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Stima parametrica

Ora ci occupiamo di come trovare un corretto stimatore, esistono due metodi, **massima verosimiglianza** e **metodo dei momenti**. Supponiamo di avere un campione statistico che dipende da un parametro $\theta$ nel quale le V.A. possono essere sia discrete che con densità.
#### Metodo della massima Verosimiglianza 
Si chiama **funzione di verosimiglianza** la funzione così definita
$$L(\theta; x_1, \dots, x_n) = \prod_{i=1}^np_{\theta}(x_1)$$
nel caso di V.A. discreta. Mentre nel caso di V.A. con densità abbiamo
$$L(\theta; x_1, \dots, x_n) = \prod_{i=1}^nf_{\theta}(x_1)$$

Questa funzione misura quanto sia probabile osservare i dati effettivamente raccolti per un fato valore del parametro. 

Da qui, la **stima di massima verosimiglianza**, se esiste, è una statistica campionaria, usualmente indicata con $\hat{\theta} = \hat{\theta}(x_1, \dots, x_n)$ tale che valga l'eguaglianza
$$L(\hat{\theta}; x_1, \dots, x_n) = \max_{\theta \in \Theta}L(\theta; x_1, \dots, x_n) \:\:\:\:\: \forall(x_1, \dots, x_n)$$
Nel caso discreto, se $x_1, \dots, x_n$ sono gli esiti del campione, la stima di massima verosimiglianza scegli il parametro $\theta$ che massimizza la probabilità degli esisti effettivamente ottenuti

#### Metodo dei Momenti
L'idea di questo metodo è di confrontare i [[Valore Atteso Varianza e Momenti|momenti teorici ]]
$$m_k(\theta) = \mathbb{E}_{\theta}[X^k]$$
con i momenti empirici(le [[Indici statistici|medie campionarie]] di $X_1^k, \dots, X_n^k)$
$$\sum_{i=1}^n\frac{x^k_i}{n}$$
Essendo che la media campionaria è un [[Stima parametrica|buono stimatore]] del valore atteso, è ragionevole prendere come stimatore $\theta$ un valore $\hat{\theta}$ che realizzi l'eguaglianza fra i momenti e la media campionaria
$$\mathbb{E}_{\hat{\theta}}[X^k] = \frac{1}{n}\sum_{i=1}^n x_i^k \:\:\:\:\:\forall(x_1, \dots, x_n)$$
# References