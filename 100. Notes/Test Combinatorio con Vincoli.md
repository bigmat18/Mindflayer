**Data time:** 14:06 - 08-04-2025

**Status**: #note #youngling 

**Tags:** [[Software Engineering]] [[Verifica e Validazione Software]]

**Area**: [[Bachelor's Degree]]
# Test Combinatorio con Vincoli

Immaginiamo 5 parametri di input <x1, x2, x3, x4, x5>
- Il dominio di x1 e x2 ripartibile i 8 classi (di cui una di valori non validi -> errore)
- Il dominio di x3 e x5 ripartibile in 4 classi (di cui una di valori non validi -> errore)
- Il dominio di x4 ripartibile in 7 classi (di cui una di valori non validi -> errore)
Se prendiamo 1 rappresentare per classe: 8 * 8 * 4 * 7 * 4 = 7168 casi di test.

Si considerano 3 strategie "a vincoli" per ridurre il numero di possibili combinazioni:
- **Vincoli di errore**
	Prendiamo l'esempio di prima <x1, x2, x3, x4, x5>. Una rappresentate per classe 8 * 8 * 4 * 7 * 4 = 7168 casi di test.

	Viene perso un solo caso, per ogni posizione, con input non valido: 5 + 7 * 7 * 3 * 6 * 3 = 2.651, quindi abbiamo ridotto a quasi 1/2 i casi.
	
- **Vincoli property/if property** 
	Definiamo dei vincoli property/if property sui primi due parametri
	- x1: classe 1, classe 2, classe 3, <u>classe 4 [property negativi]</u> classe 5, classe 6, <u>classe 7 [property positivi]</u>, <u>(classe8 [error])</u>
	- x2: classe 1, classe 3, classe 5, <u>classe 7 [if negativi]</u> classe 2, classe 4, <u>classe 6 [if positivi]</u>,  <u>(classe8 [error])</u>
	
	Quindi abbiamo 5 + (4 * 4 + 3 * 3) * 3 * 6 * 3 = 5 + 1350 = 1355 un altra importante riduzione.
	
- **Vincoli single**
	Per uno o più parametri si può decidere di testare un solo valore, per esempio applichiamo il metodo "single" ad x4 quindi abbiamo:
	5 + (4*4+3*3) * 3 * 1 * 3= 5 + 225 = 230

La tecnica basata su vincoli permette di introdurre vincoli che limitino il numero di test ottenuti dalla generazione di tutte le combinazioni di valori possibili.

Funziona bene se i vincoli che imponiamo sono **reali vincoli del dominio** e non se li aggiungiamo al solo scopo di limitare le combinazioni.
# References