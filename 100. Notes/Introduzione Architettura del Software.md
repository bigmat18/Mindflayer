**Data time:** 16:40 - 03-11-2024

**Status**: #note #master 

**Tags:** [[Architettura del software]][[Software Engineering]]

**Area**: 
# Introduzione Architettura del Software

Progettazione: architetture del software
Costituisce la fase ponte fra la specifica e la codifica, è la fase in cui si passa da “che cosa” deve essere fatto a “come” deve essere fato. Il suo prodotto si chiama architettura (o progetto) del sw.
Si hanno durante la progettazione diversi livelli di astrazione:
- Progettazione di alto livello (o architettura): Scopo è la scomposizione di un sistema in sottosistemi quindi identificazione e specifica delle parti del sistema e delle loro inter-connessioni
- Progettazione di dettaglio: decisione su come la specifica di ogni parte sarà realizzata

La dedizione di architettura software: l’architettura di un sistema software (in breve architettura software) è la strutta del sistema costituita dalle parti del sistema, dalle relazioni tra le parti, dalle loro proprietà visibili.

In altre parole l’architettura:
- definisce la struttura del sistema sw.
- specifica le comunicazioni tra componenti.
- considera aspetti non funzionali.
- è un’astrazione.
- è un artefatto complesso

## Le viste
Ci sono 3 astrazioni interessanti, che sono 3 uni di vista simultanei sul sistema sw. Vista comportamentale, vista strutturale e vista logistica.
### [[Vista comportamentale]]
Aka component-and-connector, aka C&C La vista C&C descrive un sistema software come composizione di componenti software, specifica i componenti con le loro interfacce, descrive le caratteristiche dei connettori, descrivere la struttura del sistema in esecuzione, flusso dei dati, dinamica, parallelismo, replicazioni, ... 

È utile per:  analisi delle caratteristiche di qualità a tempo d’esecuzione come prestazioni, affidabilità, disponibilità, sicurezza. Utile anche per documentare lo stile dell’architettura
### [[Viste strutturale]]
Descrive la struttura del sistema come insieme di unità di realizzazione (codice) essi: Classi, packages… A cosa serve? Analizzare dipendenze tra packages, progettare test di unità e di integrazione, valutare la portabilità.
### [[Vista logistica]]
Aka vista di deployment. Descrive l’allocazione del sw su ambienti di esecuzione. A cosa serve? Permette di valutare prestazioni e affidabilità.

![[Screenshot 2023-10-25 at 10.32.59.png | 400]]

# References