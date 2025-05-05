**Data time:** 16:55 - 03-12-2024

**Status**: #note #youngling 

**Tags:** [[Intervalli di Fiducia]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Intervalli di fiducia media campione Bernoulli

Consideriamo un campione $X_1, \dots, X_n$ di [[Variabili Aleatorie Notevoli|variabili di Bernoulli]] e cerchiamo un intervallo di fiducia per il parametro p nella forma $[\bar{X}_n \pm d]$ essendo $\bar{X}_n$ uno stimatore per p. 

Nel caso di variabili di bernoulli sappiamo che le varie $X_1, \dots, X_n$ sono di tipo Binomiale $B(n,p)$, e dunque i relativi [[Funzione di Ripartizione (CDF) e Quantili (PPF)|quantili]] che appariranno nella determinazione dell'intervallo di fiducia sono complicati e dipendono anche da n, però per n grandi possiamo usare il [[Teorema Limite di Probabilità|teorema centrale del limite]] ed approssimare
$$\frac{X_1 + \dots + X_n - np}{\sqrt{np(1-p)}} = \frac{X - np}{\sqrt{np(1-p)}}$$
Questo non è sufficiente visto che p non è nota. Però sappiamo che $\sigma^2$ è funzione del parametro p e quindi è regionevole [[Campioni Statistici e Stimatori|stimarla]] con $\bar{X}_n(1 - \bar{X}_n)$ diventando

Quindi dato un $\alpha \in (0,1)$ l'intervallo aleatorio
$$\bigg[ \bar{X}_n \pm \sqrt{\frac{\bar{X}_n(1-\bar{X}_n)}{n}} \alpha_{1-\frac{\alpha}{1}}\bigg]$$
è un intervallo di fiducia per la media p del campione $X_1, \dots, X_n$ 
# References