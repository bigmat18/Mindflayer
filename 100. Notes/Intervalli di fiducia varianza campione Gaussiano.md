**Data time:** 18:15 - 03-12-2024

**Status**: #note #youngling 

**Tags:** [[Intervalli di Fiducia]] [[Statistics]]

**Area**: [[Bachelor's Degree]]
# Intervalli di fiducia varianza campione Gaussiano

Nel caso invece si voglia trovare un intervallo di fiducia per la varianza non nota non c'è sostanzialmente differenza tra i casi in cui m è nota oppure no, consideriamo direttamente il secondo caso e ci concentriamo sul caso unilaterali che è più semplice da analizzare

Dato $\alpha \in (0,1)$ gli intervalli aleatori 
$$\bigg(0, \frac{\sum_{i=1}^n(X_i - \bar{X}_n)^2}{\chi_{\alpha,n-1}^2}\bigg] = \bigg(0, \frac{(n-1)S_n^2}{\chi^2_{\alpha, n-1}}\bigg], \:\:\: \bigg[\frac{\sum_{i=1}^n (X_i - \bar{X})^2}{\chi^2_{1-\alpha,n-1}}, +\infty\bigg) = \bigg[\frac{(n-1)S^2}{\chi^2_{1-\alpha, n-1}}, +\infty\bigg)$$
sono intervalli di fiducia per $\sigma^2$ con livelli di fiducia $1 - \alpha$
# References