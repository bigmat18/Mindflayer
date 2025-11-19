**Data time:** 01:33 - 11-05-2025

**Status**: #note #master 

**Tags:** [[Introduction to Artificial Intelligence]] [[Agenti che apprendono (ML)]] [[Introduction to Machine Learning]] [[Machine Learning]]

**Area**: [[Bachelor's Degree]] [[Master's degree]]
# Validation

Evaluation of performances for ML system is equal to generalization/predictive accuracy evaluation
> The performance on training data provide an overoptimistic evaluation

After models training on the training set:
- **Model selection**: estimating the performance (generalisation error) of different learning models in order to choose the best one (to generalise)
	- this includes search the best hype-parameters of your model (eg polynomial order,  ...). **It returns a model**
- **Model assessment**: having chosen a final model, estimating/evaluating its prediction error/risk (generalisation error) on new test data (measure of the quality/performance of the ultimately chosen model). **It return an estimation**

**Important rule**: keep separation between goals and use separate data sets.

In ad ideal work the validation use the following things:
- a large training set (to find the best hypothesis, see the theory)
- a large validation set for model selection
- a very large external unseen data for test set

With finite and often small data sets we have just an estimation of the generalisation perfromance.
## Hold out
È un modo con il quale si separano i dati in una serie di sotto-insiemi diversi, per poi essere utilizzati in fasi diverse sul modello.
- **TR** = training set. Usato per addestrare il modello
- **VL** = validation set. Da usare dopo il training, il risultato sarà confrontato fra i vari modelli per scegliere il migliore
- **TS** = test set. Usato per stimare l'errore di generalizzazione e valutare il modello

![[Screenshot 2025-11-19 at 16.32.16.png | 500]]

#### TR/VL/TS by a Schema

![[Screenshot 2025-11-19 at 16.32.56.png | 550]]

## K-fold
È un approccio che si utilizza quando abbiamo un insieme di dati limitato che deve essere usato sia per fare validation che training. 
**Convalida incrociata k-fold**:
1. Suddividere il set di dati D in k sottoinsiemi mutualmente esclusivi $D_1, D_2, \dots, D_k$
2. Addestrare l'algoritmo di apprendimento su $D \setminus D_i$  e testarlo su $D_i$ 
3. Riassumere la media di tutti i risultati $D_i$
4. Utilizzare tutti i dati per il training, la validation o test.

![[Screenshot 2024-05-18 at 21.17.20.png | 300]]
##### Issues
- How many folds? 3-fold, 5-fold, 10-fold, ....
- Often computationally very expensive
- Combinable with validation set, double-K-fold CV, ...

## Classification Accuracy

![[Screenshot 2025-11-19 at 16.35.20.png | 500]]

**Accuracy**: is the % of correctly classified patterns = TP + TN / total
Note that for binary classification: 50% correctly classified = 'coin' (random guess) prediction.

## ROC Curve
![[Screenshot 2025-11-19 at 16.37.11.png | 450]]

**ROC Curve**: the diagonal corresponds to the worst classificator. Better curves have higher AUC (Area under the curve)

## Design Cycle
##### Data collection
- Selection, integration, data cleaning etc.
- Adequately large and representative set of examples for training and test
##### Data representation
-  Domain dependent, exploit prior knowledge of the application expert 
- Feature selection
- Outliers detection
- Other preprocessing: variable scaling, missing data,.. 
Often the most critical phase for an overall success!
##### Model choice
- Statement of the problem
- Hypothesis formulation
	- You must know the limits of applicability of your model 
- complexity control
##### Building of the model (core of ML)
- through the learning algorithm using the training data
##### Evaluation
- Performance = predictive accuracy ! 
- Also interpreation of model outcomes and explanation of the results 
- “knowledge” extraction

![[Screenshot 2025-11-19 at 16.41.13.png | 200]]
# References