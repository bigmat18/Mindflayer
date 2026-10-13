---
Area: "[[Master's Degree.base]]"
Github:
Other Link: "[[Programming tools for parallel and distributed systems]]"
---
# Asymmetric and Heterogeneous Co-Processing in Path Tracing

## Contesto
L'algoritmo di **Path Tracing** è una tecnica di sintesi d'immagini basata sulla fisica che simula il trasporto della luce nell'ambiente. A differenza dei vecchi modelli di ray tracing classico (che a ogni rimbalzo generavano un albero esponenziale di raggi secondari), il path tracing adotta un approccio Monte Carlo iterativo a cammino singolo:
1. Per ogni singolo pixel dell'immagine vengono generati molteplici campioni indipendenti. Ciascun campione segue un percorso casuale autonomo.
2. Ciascun campione traccia un unico raggio primario verso la scena. Quando il raggio interseca una superficie, non si sdoppia: sceglie probabilisticamente una sola nuova direzione coerente con il materiale colpito e continua la sua traiettoria. Il raggio produce quindi una sequenza lineare di rimbalzi, non una struttura ad albero.
3. A ogni contatto, il raggio raccoglie le proprietà cromatiche della superficie, attenuando progressivamente la propria energia luminosa.
4. Il percorso di un raggio si interrompe quando si verifica una di queste condizioni:
    - Colpisce direttamente una sorgente di luce (trasportando l'energia raccolta fino alla telecamera).
    - Si disperde nello sfondo o nel vuoto senza toccare nulla.
    - Raggiunge un limite prefissato di rimbalzi massimi.
    - Viene interrotto probabilisticamente dalla Russian Roulette, un criterio statistico che arresta anticipatamente i raggi che hanno ormai perso quasi tutta la loro energia, evitando calcoli inutili.
5. Il valore cromatico del pixel è semplicemente la media stocastica dei contributi energetici raccolti da tutti i campioni calcolati per quel punto.

**References**:
- [[Ray Tracing in One Weekend.pdf]]
- [[Ray Tracing The Next Week.pdf]]
- [[Ray Tracing The Rest of Your Life.pdf]]

## Problema
Il path tracing presenta specifiche criticità intrinseche alla sua formulazione matematica e algoritmica, che lo rendono un banco di prova ideale per lo studio del calcolo eterogeneo ad alte prestazioni:
- **Disomogeneità stocastica:** Il costo computazionale è altamente non uniforme nello spazio dell'immagine. Pixel orientati verso il vuoto o geometrie aperte convergono in 1–2 rimbalzi, mentre aree con superfici riflettenti, traslucide o cavità complesse innescano catene di calcolo profonde e divergenti.
- **SIMD Divergence e dispersione di memoria:** Sui processori vettoriali (GPU), i rimbalzi stocastici disperdono rapidamente i raggi in direzioni incoerenti. Questo abbatte la località dei dati nelle cache e costringe le unità di calcolo SIMD a serializzare l'esecuzione quando thread adiacenti colpiscono materiali con proprietà fisiche differenti.
- **Latenza di coda (Tail Latency):** Quando il frame viene distribuito in parallelo, i blocchi di pixel a maggior costo stocastico finiscono per monopolizzare la fase finale dell'elaborazione. Gran parte delle unità di calcolo rimane inattiva in attesa che l'ultimo blocco completi i propri rimbalzi.

## Possibili Soluzioni
#### Opzion 1: Coordinamento Eterogeneo
1. **Decomposizione in Tile 2D:** Suddivisione dell'immagine in regioni rettangolari indipendenti, inserite in una coda concorrente sicura su CPU host gestita da oneTBB.
2. **Canale dGPU:** Invio di macro-chunk di tile alla GPU discreta per saturarne le Compute Unit e ammortizzare l'overhead di comunicazione sul bus PCIe, gestendo i buffer in modo asincrono.
3. **Canale iGPU:** Mappatura diretta della memoria RAM di sistema dell'host eliminando le copie su bus ed elaborando micro-tile flessibili a latenza di avvio ridotta.

**References**:
- **O. G. Hernández et al. - Device Specialization in Heterogeneous Multi-GPU Environments**: Formalizza la cooperazione tra GPU con diverse topologie di memoria, dimostrando i benefici dell'accoppiamento tra dGPU (high-throughput, bus-bound) e iGPU (zero-copy host memory) per task elastici a bassa latenza.
- **B. C. Budge et al. - Out-of-Core Data Management for Path Tracing on Hybrid Resources**: Uno dei lavori pionieristici sull'integrazione di risorse CPU e GPU per il path tracing, che evidenzia sperimentalmente le criticità legate all'overhead di coordinamento e alla latenza del bus PCIe.

#### Opzion 2: Gestione dello Scheduling Intelligente
Utilizzo di tecniche inteligenti per lo scheduling di task fra dGPU e iGPU:
- **Guided Chunk Decay:** La dimensione dei lotti assegnati alla dGPU non è statica, ma si riduce progressivamente al diminuire dei task residui nella coda centrale. Gli ultimi compiti inviati alla dGPU sono sufficientemente piccoli da prevenire blocchi sproporzionati a fine frame.
- **Pre-Pass a 1 SPP:** Esecuzione di una passata rapida a singolo campione prima del rendering ad alto campionamento. L'host ricava una mappa di costo (basata sulla profondità media dei raggi) e ordina i tile per entropia decrescente: i compiti pesanti vengono assegnati per primi alla dGPU, neutralizzando l'effetto straggler.
- **Host-Side Task Splitting:** Se l'iGPU conclude i propri compiti mentre la dGPU è occupata, l'host frammenta istantaneamente i macro-tile non ancora inviati al driver e li distribuisce all'iGPU in zero-copy.

**References**:
- **C. D. Polychronopoulos, D. J. Kuck (1987) - Guided Self-Scheduling: A Practical Scheduling Scheme for Parallel Supercomputers**: Il paper seminale che ha introdotto il Guided Self-Scheduling (GSS) nel calcolo parallelo. Definisce la formula matematica con cui la dimensione dei chunk inviati ai worker decresce esponenzialmente in funzione del numero di iterazioni residue
- **B. Cosenza, C. Dachsbacher, U. Erra - GPU Cost Estimation for Load Balancing in Parallel Ray Tracing**: Studio incentrato sulla non-uniformità stocastica nel ray tracing, che introduce l'impiego di mappe di costo per contrastare il disequilibrio generato dalla complessità dei materiali e delle geometrie.
- **M. H. Chung et al. - From Reactive to Proactive Load Balancing for Task-Based Parallel Runtimes**: Analizza i limiti del work-stealing reattivo in sistemi ad alta asimmetria computazionale e descrive strategie proattive per mitigare i tempi morti nelle code di esecuzione.

#### Option 3:  Struttura di Accelerazione
Sostituire le sfere analitiche con oggetti poligonali tridimensionali complessi (formati da migliaia di triangoli) o aumentare sensibilmente la quatità di strutture semplici. Per evitare di confrontare ogni raggio con ogni triangolo, i poligoni vengono organizzati in un albero gerarchico di scatole delimitatrici (**Bounding Volume Hierarchies** oppure **Linear BVH**). Sfruttare la CPU con openTBB per costruire e tenere aggiornato il tree.

**References**:
- **T. Karras (2012) - Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees**: Il paper fondamentale che ha rivoluzionato le strutture di accelerazione parallele. Spiega come generare un albero binario completo in parallelo una volta ordinate le primitive secondo i codici di Morton (curve Z-order)
- **C. Lauterbach, M. Garland, S. Sengupta et al. (2009) - Fast BVH Construction on GPUs**: Il primo paper a formalizzare la Linear Bounding Volume Hierarchy (LBVH), evidenziando il trade-off esatto tra velocità di ordinamento lineare e qualità di attraversamento rispetto al classico Surface Area Heuristic (SAH).
- **M. Pharr, W. Jakob, G. Humphreys - Physically Based Rendering: From Theory to Implementation (PBRT), 4ª Edizione (2023), Capitolo 7: Primitives and Intersection Acceleration**: Descrive nel dettaglio il layout in memoria di un array lineare di nodi BVH (ogni nodo memorizza i limiti AABB e l'offset dell'indice del figlio destro/primitiva foglia) e la funzione di attraversamento iterativa non ricorsiva (con stack fisso locale a 32-64 interi).

#### Option 4:  Wavefront Path Tracing
Abbandonare il singolo kernel monolitico che segue il raggio dalla partenza alla fine. Il processo viene spezzato in kernel distinti (generazione raggi, intersezione con la scena, valutazione dei materiali) che comunicano leggendo e scrivendo su code di memoria condivise. Elimina del tutto la divergenza SIMD (tutti i thread di un blocco eseguono la stessa identica operazione), introducendo però la necessità di compattare continuamente i dati in memoria globale GPU per scartare i raggi terminati.

**References:**
- **S. Laine, T. Karras, T. Aila (2013) - Megakernels Considered Harmful: Flexible Ray Tracing on GPUs**: L'articolo seminale (prodotto dai ricercatori NVIDIA) che ha introdotto il paradigma Wavefront per GPU. Dimostra quantitativamente come i megakernel monolitici causino una perdita di efficienza dovuta alla SIMD divergence e alla pressione sui registri, proponendo la separazione in kernel specializzati, collegati da code di raggi.
- **B. Burley et al. (Disney Studios, 2018) - The Design and Evolution of Disney's Hyperion Renderer**: Racconta la transizione industriale di Disney verso un motore di produzione basato interamente su percorsi di raggi raggruppati, ordinati e smistati in streaming, validando l'architettura a code su scene complesse con decine di milioni di rimbalzi.
- **M. Pharr, W. Jakob, G. Humphreys - Physically Based Rendering (PBRT-v4) (2023), Capitolo 15: Wavefront Rendering on GPUs**: La quarta edizione di PBRT documenta la riscrittura del loro intero sottosistema GPU passando dal megakernel all'architettura Wavefront. Spiega la gestione delle code Structure of Arrays (SoA), le primitive di compattazione stream parallela e la sincronizzazione tra fasi di calcolo.


## Efficienza Energetica
##### Architetture Mobile e Path Tracing Low-Power
Questo filone analizza come eseguire il trasporto della luce su sistemi con budget termici ristretti (5W–15W) e alimentazione a batteria:
- **D. Kopta, K. Shkurko, J. Spjut, E. Brunvand, A. Davis (2015)** — _«An Energy-Efficient Architecture for Mobile Ray Tracing»_ : Uno dei lavori accademici fondamentali sull'impatto energetico del ray tracing su architetture mobili. Dimostra che l'energia spesa per frame è dominata dagli accessi alla memoria globale (DRAM) causati dai raggi incoerenti, e non dai calcoli matematici delle ALU.
    - **Cosa prendere:** La metodologia di profiling energetico: dimostrare come il partizionamento dei dati serva a massimizzare la località nelle cache per non prosciugare la batteria attraverso il memory controller.

- **J. Spjut, D. Kopta, R. Bowles, E. Brunvand, A. Davis (2012)** — _«TRaX: A Dual-Threaded Processor Architecture for Real-Time Ray Tracing in Handheld Devices»_: Esplora il design di architetture di calcolo parallele dedicate al rendering su dispositivi portatili (_handheld_), introducendo metriche di Throughput-per-Area e Throughput-per-Watt.

##### Scheduling Eterogeneo con Power Capping e DVFS
Questi paper affrontano il bilanciamento di carico non per massimizzare la velocità pura, ma per operare sotto un tetto massimo di Watt (_Power Budget_):
- **Y. Cho, S. Hong, H. Kim (2018)** — _«Energy-Efficient Heterogeneous Computing under Performance and Thermal Constraints»_: Studia lo scheduling cooperativo tra CPU multi-core e GPU integrata su SoC mobili. Dimostra che forzare entrambi i processori al massimo delle frequenze innesca il _Thermal Throttling_ (DVFS abbassa i clock bruscamente), mentre ripartire asimmetricamente i compiti mantenendo le frequenze medie riduce l'EDP (_Energy-Delay Product_).
    - **Cosa prendere:** L'impostazione sperimentale: puoi testare il tuo path tracer a 15W o 25W costanti e misurare se il tuo _Guided Chunk Decay_ previene i picchi di temperatura rispetto a un partizionamento ingenuo.

- **V. W. Freeh, F. Pan, N. Kappiah, D. K. Lowenthal (2007)** — _«Exploring the Energy-Performance Tradeoff for Heterogeneous High-Performance Computing»_ (IEEE IPDPS).
    - **Cosa tratta:** Studio formale sul trade-off tra tempo di esecuzione ed energia assorbita quando si distribuisce il carico su worker con diverse efficienze energetiche, formalizzando il punto in cui l'accelerazione non giustifica più l'energia marginale spesa.

##### Contesa di Memoria su Architetture UMA (Unified Memory Architecture)
Nei sistemi Edge e nei SoC moderni (SoC ARM, laptop ultraleggeri, APU), CPU e GPU condividono la stessa memoria fisica (LPDDR4/LPDDR5). Questo introduce colli di bottiglia diversi rispetto al bus PCIe:
- **S. Ghose et al. (2018)** — _«Demystifying Complex Workload Behavior on Integrated CPU-GPU Systems»_: Quantifica empiricamente l'interferenza tra CPU e GPU integrata quando operano simultaneamente sulla RAM di sistema. Mostra che se la GPU integrata esegue accessi casuali alla memoria (come nel path tracing), degrada la banda disponibile per la CPU host fino al 40%.
    - **Cosa prendere:** È la giustificazione scientifica per misurare il traffico di memoria host quando la iGPU lavora in Zero-Copy: dimostrare se il partizionamento a micro-chunk evita la saturazione del memory controller.

##### Ray Tracing per Edge Computing e Sensori Autonomi (NVIDIA Jetson / Automotive)
Nell'ultimo lustro il ray tracing su piattaforme embedded è diventato cruciale per la robotica e i veicoli autonomi, dove sensori ottici, radar e LiDAR vengono simulati in tempo reale direttamente sul nodo a bordo:
- **P. P. Vázquez, E. Cordon (2020)** — _«Real-Time Sensor Simulation on Low-Power Embedded Devices»_: Valuta l'uso del ray tracing su piattaforme embedded (famiglia NVIDIA Jetson a 10W–30W) per simulare dati sensoriali sintetici in tempo reale a basso consumo energetico.
    - **Cosa prendere:** Il caso d'uso applicativo da inserire nella sezione _Motivazione_ del report: spiegare che lo studio di un path tracer asimmetrico a basso consumo trova applicazione diretta nei sensori di navigazione autonoma on-edge.

# References

