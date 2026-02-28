---
Data: 2026-02-27T13:40:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Univariate Optimization]]"
Area: "[[Master's degree]]"
---
# A Fleeting Glimpse to Global Optimization

Tutto quello che abbiamo visto finora riguarda la ricerca di minimi locali. Ma cosa ci dice tutto questo sull'ottimizzazione globale?
Purtroppo, la risposta è: **quasi nulla, a meno che non vengano fatte assunzioni forti**.

Il problema fondamentale delle funzioni non convesse è che possono presentare molteplici minimi locali. Quando un algoritmo converge in una "buca", non ha modo di sapere se esiste una buca più profonda da qualche altra parte. Intuitivamente, vogliamo evitare la situazione in cui un punto stazionario sia solo un minimo locale e non globale. 
La condizione sufficiente per far sì che ogni minimo locale sia anche globale è che $f'(x) \ge 0$ per tutto il dominio, ovvero che la funzione sia **convessa**.

## A Very Quick Glimpse to Convexity

L'idea alla base della convessità è che la funzione ha una forma a "ciotola" rivolta verso l'alto. 
In termini matematici:
- Convessità $\simeq f'$ è monotona non decrescente $\simeq f'' \ge 0$.

Attenzione però: questa è un'intuizione legata alla derivabilità. In realtà, una funzione convessa non deve necessariamente appartenere a $C^1$ (e ancor meno a $C^2$); ad esempio, la funzione valore assoluto $f(x) = |x|$ è convessa ma presenta un punto non derivabile.

Il mondo delle funzioni convesse è relativamente vasto e possiede ottime proprietà:
- Alcune funzioni base sono intrinsecamente convesse e molte operazioni matematiche preservano questa convessità, permettendo di costruire insiemi e funzioni multivariate complesse.
- Esiste un'enorme mole di teoria e software dedicato a risolverle.
- Molti modelli di Machine Learning (come le Support Vector Machines, SVM) sono costruiti di proposito per essere convessi, in modo che trovare l'ottimo globale sia "facile" e garantito.

La regola d'oro nell'ottimizzazione è: **"Se hai la possibilità di scegliere, scegli un modello convesso"**.
Ma cosa succede se il problema è intrinsecamente non convesso e hai assoluta necessità di trovare l'ottimo globale?

## The Spatial Branch-and-Bound Approach

Se non possiamo usare la discesa del gradiente o il metodo di Newton per trovare il minimo globale, dobbiamo ispezionare l'intero dominio $X = [x_-, x_+]$, ma facendolo in modo intelligente (evitando la ricerca cieca).

L'approccio prevede la costruzione di una **approssimazione convessa dal basso** (lower approximation) $\underline{f}$ della nostra funzione originaria non convessa $f$ sull'intervallo $X$.

Poiché $\underline{f}$ è convessa per definizione, è "facile" calcolarne il minimo locale (che coinciderà con il suo minimo globale). Chiamiamo questo punto $\overline{x}$. 
Questo ci fornisce un'informazione cruciale, ovvero un limite inferiore e uno superiore per il vero minimo globale $f_*$:
$$\underline{f}(\overline{x}) \le f_* \le f(\overline{x})$$

*L'algoritmo valuta la bontà di questa approssimazione misurando il "gap", ovvero la differenza tra il valore reale $f(\overline{x})$ e il valore approssimato $\underline{f}(\overline{x})$. Se questo gap è troppo grande (superiore a una tolleranza), l'algoritmo partiziona l'intervallo originario $X$ in sotto-intervalli più piccoli e reitera il processo su ciascuno di essi. Poiché l'approssimazione dal basso dipende dalla larghezza della partizione, intervalli più piccoli genereranno un'approssimazione più fedele e, di conseguenza, un gap minore. Il vero vantaggio si ha nel "pruning" (taglio): se durante la ricerca si scopre che il minimo limite inferiore $\underline{f}(\overline{x})$ di una certa partizione è maggiore o uguale al miglior valore reale di $f$ trovato finora, quella partizione viene scartata definitivamente ("killed for good"), garantendo che lì dentro non si nasconda alcun minimo globale.*

## Is Something Like This Efficient?

In una parola? **Sicuramente no** nel caso peggiore (worst-case scenario).
Il rischio di dover continuare a tagliare e affettare $X$ finché i pezzi non diventano minuscoli porta inevitabilmente a un tempo di calcolo **esponenziale**.

Tuttavia, nella pratica l'efficienza dipende fortemente da due fattori:
1. Quanto è "fortemente non convessa" la funzione $f$ reale.
2. Quanto è stretta e accurata l'approssimazione $\underline{f}$.

Esistono approcci intelligenti che cercano di isolare e gestire solo le fonti di non-convessità. I problemi si dividono in classi di difficoltà crescente:
- **Mixed-Integer Linear Programs (MILP):** Tutto diventa banale se si fissano o si rilassano le variabili intere (che sono la fonte della non-convessità). Rimane comunque un problema a tempo esponenziale nel caso peggiore.
- **Mixed-Integer Nonlinear Convex Programs (MINLP):** Concettualmente ancora "semplici" da inquadrare, sebbene numericamente più difficili da risolvere. Ancora esponenziali.
- **(Mixed-Integer) Nonlinear Nonconvex Programs:** Questa è la classe più difficile, dove trovare una qualsiasi approssimazione $\underline{f}$ è molto complesso. La strategia tipica consiste nel riscrivere l'espressione di $f$ scomponendola in funzioni unarie/binarie elementari e applicare specifiche formule di convessificazione per ciascuna. Il problema è che, sebbene questo processo sia tecnicamente efficiente da eseguire, molto spesso genera un'approssimazione $\underline{f}$ pessima (limiti deboli), portando nuovamente a un'esplosione combinatoria esponenziale.

La buona notizia è che questi complessi metodi di convessificazione e bounding sono già implementati in solver commerciali e open-source estremamente ben ingegnerizzati. Nella pratica, usare questi solver è immensamente meno inefficiente rispetto al provare punti a caso nel dominio (blind search). 

L'amara verità, però, è che rimangono pur sempre **immensamente meno efficienti** rispetto agli algoritmi di ottimizzazione locale visti in precedenza.