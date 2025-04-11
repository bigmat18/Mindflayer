**Data time:** 14:04 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: 
# Metodo statistico

I casi di test sono selezionati in base alla distribuzione di probabilità dei dati di ingresso del programma. Il test è quindi progettato per esercitare il programma sui valori di ingresso più probabili per il suo utilizzo a regime. Il vantaggio è che, nota la distribuzione di probabilità, la generazione dei dati di test è facilmente automatizzabile. Non sempre corrisponde alle effettive condizioni d’utilizzo del software, è oneroso calcolare il risultato atteso (problema dell’oracolo).

**Esempio**.
Si considera l'input "età il giorno della laurea" (il tipo è int). In questo caso è ragionevole usare il metodo statistico e dare la specifica di test (Test obligation):
- tutti i valori compresi tra 20 e 27,
- il 40% dei valori tra 27 e 35 (questi possono essere scelti in modo random),
- il 5% dei valori tra 36 e 100 (Questi possono essere scelti in modo random). 
Casi di test che soddisfano le test obligations: <20, _, _>,  <21, _, _>, ..., <27, _, _>, <29, _, _>, ... <51, _, _> ... (al momento non sono ancora specificati output e ambiente)

# References