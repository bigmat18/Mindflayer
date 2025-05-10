**Data time:** 00:51 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Progettazione Basi di Dati]]

**Area**: [[Bachelor's Degree]]
# Associazioni 1-1

Le associazioni uno a uno si rappresentano aggiungendo la chiave esterna ad una qualunque delle due relazioni che riferisce l’altra relazione, preferendo quella rispetto a cui l’associazione è totale, nel caso in cui esista un vincolo di totalità.
![[Screenshot 2023-11-22 at 20.31.51.png]]

La direzione dell’associazione rappresentata dalla chiave esterna è detta “la diretta” dell’associazione. Ci sono alcuni vincoli sulla cardinalità delle associazioni uno a molti ed uno ad uno:
- Univocità della diretta.
- Totalità della diretta: si rappressenta imponendo un vincolo not null sulla chiave esterna.
- Univocità dell'inversa e totalità della diretta: si rappresenta imponendo un vincolo not null ed un vincolo di chiave sulla chiave esterna.

![[Screenshot 2023-11-22 at 20.34.16.png]]

# References