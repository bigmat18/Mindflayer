**Data time:** 00:53 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Progettazione Basi di Dati]]

**Area**: [[Bachelor's Degree]]
# Partizione Verticale in Modello Relazionale

La gerarchia si trasforma in due associazioni uno a uno che legano rispettivamente la classe genitore con le classi figlie. In questo caso non c’è un trasferimento di attributi o di associazioni e le classi figlie $A_1$ ed $A_2$ sono identificate esternamente dalla classe genitore $A_0$. 

Nello schema ottenuto vanno aggiunti dei vincoli: ogni occorrenza di $A_0$ non può partecipare contemporaneamente alle due associazioni, e se la gerarchia è totale, deve partecipare ad almeno una delle due.

**Esempio**
Classe Corsi con due attributi Codice (la chiave), Nome e con due sottoclassi di tipo partizione: CorsiInterni, con un attributo Crediti, CorsiEsterni, con due attributi CorsoDiLaurea, AnnoAccademico.

![[Screenshot 2023-11-22 at 20.51.17.png]]

![[Screenshot 2023-11-22 at 20.51.35.png]]

# References