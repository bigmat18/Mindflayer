**Data time:** 16:03 - 10-05-2025

**Status**: #note #youngling 

**Tags:** [[Basi di Dati]] [[Modello Relazionale]]

**Area**: [[Bachelor's Degree]]
# Vincoli Referenziali

Nel [[Introduzione Modello Relazionale|modello relazionale]] le informazioni in relazioni diverse sono correlate attraverso valori comuni, in particolare, vengono spesso presi in considerazione i valori delle [[Chiavi nel Modello Relazionale|chiavi]] (primarie). Le correlazioni debbono essere "coerenti".
![[Screenshot 2023-11-22 at 19.21.31.png]]

*Definizione*: un vincolo di **integrità referenziale** o (**foreign key**) fra gli attributi $X$ di una relazione $R_1$ e un'altra relazione $R_2$ impone ai valori su $X$ in $R_1$ di comparire come valori della chiave primaria $R_2$.
![[Screenshot 2023-11-22 at 19.25.55.png]]       ![[Screenshot 2023-11-22 at 19.25.34.png]]
**Esempio**. Vincoli di integrità referenziale fra:
- l’attributo Vigile della relazione INFRAZIONI e la relazione VIGILI
- gli attributi Stato e Numero di INFRAZIONI e la relazione AUTO

**Vincoli multipli su più attributi**
![[Screenshot 2023-11-22 at 19.27.08.png]]
Vincoli di integrità referenziale fra:
- La coppia di attributi StatoA e NumeroA di INCIDENTI e la relazione AUTO
- La coppia di attributi StatoB e NumeroB di INCIDENTI e la relazione AUTO
# References