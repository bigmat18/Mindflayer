**Data time:** 13:45 - 26-11-2024

**Status**: #note #youngling 

**Tags:** [[Statistice riassuntive e grafici]] [[Statistics]]

**Area**: 
# Percentili, Quantili e Boxplot

Consideriamo di avere un vettori di quantità numeriche $(x_1, ..., x_n) \in \mathbb{R}^n$ dato un numero $0 < k < 100$ chiediamoci come trovare un valore soglia che divida in due parti i dati in modo che una parte contenga i dati più piccoli di tutti i valori della soglia e che contenga esattamente k% dei dati.
##### K-mo Percentile
Il dato $x_i$ tale che:
- almeno $\frac{k}{100} \cdot n$ dati siano inferiori o eguali a $x_i$ 
- almeno $\frac{100 - k}{100} \cdot n$ siano superiori o eguali a $x_i$

![How Percentiles Work (and Why They're Better Than Averages) | 300](data:image/png;base64,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)
Per calcolare il k-mo percentile di parte da n numerosità del campione $x_{(1)}, ..., x_{(n)}$ ordinato in senso crescente, poi prendiamo $\beta = k / 100$ e
- se $\beta n$ non è intero si prende come k-mo percentile il valore $x_{\lceil \beta_{n} \rceil )}$ 
- se $\beta n$ è intero si prende come k-mo percentile la [[Indici statistici|media aritmetica]] tra $x_{(\beta_n)}$ e $x_{(\beta_{n+1})}$ 

#### $\beta$-Quantile
Dato un $k \in (0, 100)$  e posto $\beta = \frac{k}{100} \in (0, 1)$ il k-mo percentile è anche detto $\beta$-quantile

##### Boxplot
È una rappresentazione grafica dei dati che evidenza l'intervallo in cui sono contenuti i dati. Si ottiene sovrapponendo a una linea he va dal minimo al massimo dei dati un rettangolo che va dal primo al terzo quantile (25-mo percentile, 75-mo percentile).

![Box Plot Explained: Interpretation, Examples, & Comparison | 350](https://www.simplypsychology.org/wp-content/uploads/box-whisker-plot.jpg)

# References