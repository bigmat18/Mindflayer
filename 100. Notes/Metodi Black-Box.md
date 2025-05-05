**Data time:** 14:03 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: [[Bachelor's Degree]]
# Metodi Black-Box

Sono criteri per l'individuazione dei casi di input che si basano sulle specifiche. La strategia è al seguente:
- Separare le funzionalità da testare (per esempio usando i casi d'uso)
- Derivare un insieme di casi di test per ogni funzionalità
- M(p1, p2, p3, p4) < <i1, i2, i3, i4>, output atteso, ambiente >
	Per fare ciò bisogno: per ogni (tipo di) paramentro di input andare ad individurare dei valori da testare (per questo si usano alcune tecniche (metodi) che vediamo sotto), e per l'insieme dei parametri si usano tecniche che vanno sotto il nome di testing combinatorio per ridurre le combinazioni.

### [[Metodo statistico]]

### [[Partizione dei dati in ingresso in classi di equivalenza ]]

### [[Metodo random]]

### [[Test basato su catalogo]]

## Test combinatorio
Tecnica da applicare al crescere del numero dei parametri in input.
In presenza di più dati in input, se si prende il prodotto cartesiano dei casi di test individuati, facilmente si ottengono numeri non gestibili.
Occorrono quindi strategie per generare casi di test significativi in modo sistematico.

Ci sono 2 tecniche per ridurre **l'esplosione combinatoria**.
- **Vincoli**
- **Pairwise testing**
### [[Test Combinatorio con Vincoli]]

### [[Test Combinatorio con Pairwise Testing]]
# References