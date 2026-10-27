---
Data: 2026-10-25T15:44:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Sorting Atomic Items]]"
Area:
  - "[[Master's Degree.base]]"
---
# Sorting with Multi-Disks

The bottleneck in disk-based sorting is the time needed to perform an I/O operation. In order to mitigate this problem, we can use D disks working in parallel so as to transfer DB items per I/O.
- this **increases the [[Processing Bandwidth & Throughput|Bandwidth]]** of the I/O subsystem
- but it makes the design of **I/O-efficient algorithms particularly difficult**.

The simplest approach to managing parallel disk is called **Disk Stripping**.
> [!NOTE] Definition
> The **Disk Stripping** consists ok looking at the $D$ disks ad **one single disk** whose page size is $B' = DB$.

With this tecnique any algorithm designed for one disk can be used with a disk page size of $B'$. But on the other hang **we lose the independence among the $D$ disks**, and this comes to some price in terms of $I/O$ complexity:
$$
O\left(\frac{n}{B'} \log_{M/B'} \frac{n}{M}\right) = O\left(\frac{n}{DB} \log_{M/DB} \frac{n}{M}\right) \text{ I/O}
$$

This bound is not optimal, because the base of the logarithm we have a different from the **optimal bound**:
- Nel bound ottimale la base è $M/B$
- Nel Disk Striping la base scende a $M/DB$

Poiché la base del logaritmo si rimpicciolisce da $M/B$ a $M/DB$, il numero di passaggi completi sui dati aumenta3. Il rapporto tra il bound ottimale e quello del Disk Striping è pari a $1 - \log_{M/B} D$. 

$$
\frac{\text{disk stripping}}{\text{lower bound}} = \frac{\frac{n}{DB}\log_{M/DB}\frac{n}{M}}{\frac{n}{DB}\log_{M/B}\frac{n}{M}} = \frac{\log_{M/DB}\frac{n}{M}}{\log_{M/B}\frac{n}{M}}
$$
applichiamo il cambio di base al logaritmo denominatore:
$$\log_{M/DB} \left(\frac{n}{M}\right) = \frac{\log_{M/B} \left(\frac{n}{M}\right)}{\log_{M/B} \left(\frac{M}{DB}\right)} $$
sostituiamo e semplifichiamo:
$$\frac{\log_{M/B} \left(\frac{n}{M}\right)}{\frac{\log_{M/B} \left(\frac{n}{M}\right)}{\log_{M/B} \left(\frac{M}{DB}\right)}} = \log_{M/B} \left(\frac{M}{DB}\right) $$
Riscriviamo la frazione $\frac{M}{DB}$ mettendo in evidenza il termine $\frac{M}{B}$:
$$\frac{M}{DB} = \frac{\left(\frac{M}{B}\right)}{D} $$
Utilizzando la proprietà dei logaritmi per il quoziente ($\log(A/B) = \log A - \log B$):
$$\log_{M/B} \left(\frac{\frac{M}{B}}{D}\right) = \log_{M/B} \left(\frac{M}{B}\right) - \log_{M/B}(D) $$
Poiché il logaritmo di un numero nella sua stessa base è uguale a 1 ($\log_{M/B}\left(\frac{M}{B}\right) = 1$), otteniamo il risultato finale:
$$\text{Rapporto} = 1 - \log_{M/B} D $$
More Disk you add, worst is your algorithm because the second parte increase and the overall result decrease. Ciò dimostra che **il Disk Striping perde efficienza man mano che il numero di dischi** $D$ **aumenta** (avvicinandosi alla capacità in blocchi della RAM $M/B$
# References