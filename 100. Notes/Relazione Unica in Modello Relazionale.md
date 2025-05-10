**Data time:** 00:52 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Basi di Dati]] [[Progettazione Basi di Dati]]

**Area**: [[Bachelor's Degree]]
# Relazione Unica in Modello Relazionale

Se $A_0$ è la classe genitore di $A_1$ ed $A_2$, le classi $A_1$ e $A_2$ vengono eliminate ed accorpate ad $A_0$. Ad $A_0$ viene aggiunto un **attributo (Discriminatore)** che indica da quale delle classi figlie deriva una certa istanza, e gli attributi di $A_1$ ed $A_2$ vengono assorbiti dalla classe genitore, e assumono valore nullo sulle istanze provenienti dall’altra classe. 

Infine, una relazione relativa a solo una delle classi figlie viene acquisita dalla classe genitore e avrà comunque cardinalità minima uguale a 0, in quanto gli elementi dell’altra classe non contribuiscono alla relazione.

**Esempio**. Classe Corsi con due attributi Codice (la chiave), Nome e con due sottoclassi di tipo partizione: CorsiInterni, con un attributo Crediti, CorsiEsterni, con due attributi CorsoDiLaurea, AnnoAccademico.

![[Screenshot 2023-11-22 at 20.40.16.png]]

![[Screenshot 2023-11-22 at 20.48.57.png]]

# References