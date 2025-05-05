**Data time:** 16:47 - 03-11-2024

**Status**: #note #master 

**Tags:** [[Design Patterns]][[Software Engineering]]

**Area**: [[Bachelor's Degree]]
# Introduzione Design Patterns

Esistono una serie di regole pratiche che il progettista può seguire per costruire qualcosa. Queste regole pratiche sono i design patters, e sono definiti grazie a anni di esperienza da parte di terze persone. Si applicano in fase di design.

![[Screenshot 2023-11-15 at 12.47.22.png]]

### GOF design patterns
Sono 23 design patters suddivisi in base al loro scopo:
- **Creazionali:** propongono soluzioni per creare oggetti
- **Comportamentali:** propongono soluzioni per gestire il modo in cui vengono suddivise le responsabilità delle classi e degli oggetti.
- **Strutturali:** propongono soluzioni per la composizione strutturale di classi e oggetti.

Perché i pattern nel software? “Progettare software OO è difficile e progettare software OO riutilizzabile è ancora più difficile” cit. Erich Gamma.
I progettisti esperti riutilizzano le soluzioni che hanno funzionato in passato, così i sistemi OO ben strutturati hanno modelli ricorrenti di classi e oggetti. La conoscenza degli schemi che hanno funzionato in passato consente al progettista di essere più produttivo e ai progetti risultanti di essere più flessibili e riutilizzabili.

Ci sono vari libelli di astrazioni per i design pattern. Si passa da design pattern complessi per intere applicazioni o sottosistemi. Si passa poi a soluzioni per problemi generali in un determinato contesto, fino ad arrivare a design class più semplici come linked List, hashtag table ecc..

Nell’architettura - progettazione di dettaglio-codice differenziamo i patterns dagli stili architetturali che sono pipes and filters, publish-subscribe, model-view-controller mentre i design patterns sono per la progettazione e raffinamento dei singoli componenti. 

Possiamo poi definire gli **Idiomi** che sono pattern di basso livello specifici di un linguaggio di programmazione.
### GOF patterns tramplate (come viene proposto)
1. Nome del pattern e classificazione
2. Scopo: piccolo riassunto di cosa il pattern va a fare
3. Conosciuto anche come: altri nomi con cui viene chiamato il pattern.
4. Motivazione: illustrazione di uno scenario dove il modello potrebbe essere utile.
5. Applicabilità: Situazione in cui è possibile utilizzare il modello.
6. Struttura: rappresentazione grafica del pattern.
7. Partecipanti: le classi e gli oggetti che partecipanti al pattern
8. Collaboratori: come interagiscono i partecipanti per svolgere le loro responsabilità?
9. Conseguenze: Quali sono i pro ed i contro dell’utilizzo del modello?
10. Implementazione: Suggerimenti e tecniche per implementare il modello.
11. Un sample del codice: un frammento di codice esempio dell’implementazione.
12. Dove viene usato: Esempi di veri sistemi che usano questo pattern
13. Pattern collegati: Altri pattern collegati a questo.
### Notazione usata da GOF
Il libro GoF usa la moderazione ad oggetti per la notazioni dei diagrammi di [[Diagramma delle Classi|classi]] e [[Diagramma degli Oggetti|oggetti]]

![[Screenshot 2023-11-15 at 14.36.44.png | 400]]

![[Screenshot 2023-12-07 at 13.03.07.png | 400]]

![[Screenshot 2023-11-15 at 14.37.25.png | 400]]

![[Screenshot 2023-11-15 at 14.37.39.png ]]

![[Screenshot 2023-11-15 at 14.37.52.png | 500]]

Classi e istanze (metadata & data)
![[Screenshot 2023-11-15 at 14.38.03.png | 500]]
# References