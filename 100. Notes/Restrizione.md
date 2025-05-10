**Data time:** 16:11 - 10-05-2025

**Status**: #note #master

**Tags:** [[Basi di Dati]] [[Operatori Insiemistici]] [[Algebra Relazionale]]

**Area**: [[Bachelor's Degree]]
# Restrizione

Restrizione (Selezione): $\sigma_{condizione}(R)$ 

**Esempi**:
- $\sigma_{Nome = 'Caio'}(Studenti)$
- **Composizione di operatori**: $\pi_{Matricola}(\sigma_{Nome = 'Caio'}(Studenti))$ 

![[Screenshot 2023-11-26 at 17.49.41.png]]

Un risultato non desiderabile:
$$\sigma_{età > 30}(Persona) \cup \sigma_{Età \leq 30}(Persone) \neq Persone$$
Questo perchè le sezioni vengono valutare separatamente. Ma anche:
$$ \sigma_{età > 30 \lor età \leq30} (Persone) \neq Persone $$
Perché anche le condizioni atomiche vengono valutate separatamente.
La condizione atomica è inoltre vera solo per valori non nulli, per riferirsi ai valori nulli esistono forme apposite di condizioni: IS NULL, IS NOT NULL.

A questo punto:
$$\sigma_{età > 30}(Persona) \cup \sigma_{Età \leq 30}(Persone) \cup \sigma_{Eta \: IS \: NULL}(Persone) = $$
$$= \sigma_{età > 30 \lor età \leq 30 \lor Età \: IS \: NULL} (Persone) = Persone$$
**Esempio**.
![[Screenshot 2023-11-26 at 18.09.15.png]]

# References