**Data time:** 00:45 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Normalizzazione]]

**Area**: [[Bachelor's Degree]]
# Derivazione

*Definizione*: Sia F un insieme di DF, diremo che X → Y sia derivabile da F (F |– X → Y), se X → Y può essere inferito da F usando gli assiomi di Armstrong.

Si dimostra che valgono le regole:
- {X → Y, X → Z} |– X → YZ (unione U) 
- Z $\subset$ Y {X → Y} |– X → Z (decomposizione D) 
- L’unione: {X → Y, X → Z} |– X → YZ (unione U)
	1. X → Y (per ipotesi) 
	2. X → XY (per arricchimento da 1) 
	3. X → Z (per ipotesi) 
	4. XY → YZ (per arricchimento da 3) 
	5. X → YZ (per transitività da 2, 4)

*Definizione*: Una derivazione di f da F è una sequenza finita f1 , …, fm di dipendenze, dove fm = f e ogni fi è un elemento di F oppure è ottenuta dalle precedenti dipendenze f1 , …, fi-1 della derivazione usando una regola di inferenza.

Si dimostrano che valogono anche le regole:
- {X → Y, X → Z} |– X → YZ (unione U) 
- Z $\subseteq$ Y {X → Y} |– X → Z (decomposizione D) 
Da Unione e Decomposizione si ricava che se Y = A1A2…An allora:
- X → Y $\Leftrightarrow$ {X → A1 , X → A2 , …, X → An }

*Teorema*: Gli assiomi di Armstrong sono corretti e completi.

Attraverso gli assiomi di Armstrong, si può mostrare l’equivalenza della nozione di implicazione logica (|=) e di quella di derivazione (|-): se una dipendenza è derivabile con gli assiomi di Armstrong allora è anche implicata logicamente (correttezza degli assiomi), e viceversa se una dipendenza è implicata logicamente allora è anche derivabile dagli assiomi (completezza degli assiomi).

**Correttezza** degli assiomi: $\forall$ f, F |- f $\Rightarrow$ F |= f
**Completezza** degli assiomi: $\forall$ f, F |= f $\Rightarrow$ F |- f

# References