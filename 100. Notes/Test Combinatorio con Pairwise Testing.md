**Data time:** 14:07 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: 
# Test Combinatorio con Pairwise Testing

Viene anche chiamato combinazione di test basato su coppie.
Prendiamo sempre foo(x1, x2, x3, x4, x5). Nel caso in cui il dominio non contenga in sé dei vincoli è preferibile optare per un'altra tecnica: **la generazione di tutte le combinazioni solo per i sottoinsiemi di 2 variabili**

In genere si possono generare tutte le combinazioni solo per i sottoinsiemi di k variabili con k < n (pairwise quando k = 2).

L'idea: generare tutte le possibili combinazioni solo per **possibili** coppie di variabili.

**Esempio**. Riprendiamo il caso <x1, x2, x3, x4, x5>. Tutte le combinazione portano a 8 * 8 * 4 * 7 * 4 = 7.168. Quanto si risparmia con il pairwise ?
8 * 8+ 8 * 4 + 8 * 7 + 8 * 4  +        8 * 4 + 8 * 7 + 8 * 4 +       4 * 7 + 4 * 4 + 7 * 4 = 371, in realtà sono anche meno se generiamo le combinazioni in maniera efficiente.

**Esempio**.
![[Screenshot 2023-12-04 at 12.20.14.png]]
Se volessimo generare tutte le combinazioni per Display mode, screen size e fonts avremo $3^3 = 27$. Andiamo dunque a generare tutte le combinazioni per la coppia <display mode, screen size>, abbiamo quindi $3^2 = 9$.
Poi occorre generare anche tutte le combinazioni per le coppie FontsxScreen e FontsxDisplay. 
Ma in questo caso generando le combinazioni per la prima coppia il valore del terzo parametro può essere aggiunto in modo da coprire anche tutte le combinazioni di FontsxScreensize e FontsxDispay mode.

![[Screenshot 2023-12-04 at 12.22.44.png | 400]]

La generazione di combinazioni che in maniera efficiente coprano tutte le coppie è impossibile da fare a mano per molti parametri con molti valori ma può essere fatta con euristiche.

# References