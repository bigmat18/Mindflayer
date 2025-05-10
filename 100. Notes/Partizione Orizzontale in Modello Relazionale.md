**Data time:** 00:53 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Progettazione Basi di Dati]]

**Area**: [[Bachelor's Degree]]
# Partizione Orizzontale in Modello Relazionale

La classe genitore $A_0$ viene eliminata, e le classi figlie $A_1$ ed $A_2$ ereditano le proprietà (attributi, identificatore e relazioni) dell’classe genitore.

**Esempio**. Classe Corsi con due attributi Codice (la chiave), Nome e con due sottoclassi di tipo partizione: CorsiInterni, con un attributo Crediti, CorsiEsterni, con due attributi CorsoDiLaurea, AnnoAccademico.

![[Screenshot 2023-11-22 at 20.47.46.png]]

Il partizionamento orizzontale divide gli elementi della superclasse in più relazioni diverse, per cui non è possibile mantenere un vincolo referenziale verso la superclasse stessa; in conclusione, questa tecnica non si usa se nello schema relazionale grafico c’è una freccia che entra nella superclasse.

![[Screenshot 2023-11-22 at 20.49.23.png]]

# References