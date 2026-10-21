---
Data: 2026-09-29T20:24:00
Tags:
  - note
  - youngling
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# Boxes and Hero
Il problema chiede di selezionare un sottoinsieme di $n$ scatole (ciascuna con peso $w_i$ e resistenza $s_i$) per formare la torre più alta possibile (massimizzando il numero di scatole). Ogni scatola $i$ nella torre deve soddisfare il vincolo:
$$s_i \ge \sum_{\text{scatola } j \text{ sopra } i} w_j$$
### L'ordinamento canonico (Exchange Argument)
Supponiamo di aver già scelto un sottoinsieme valido di scatole. In che ordine verticale devono stare per rendere il vincolo il più permissivo possibile? Prendiamo due scatole adiacenti nella torre: la scatola $A$ sopra la scatola $B$.

Sia $W$ il peso totale di tutte le scatole situate sopra $A$.
- Per $A$: $s_A \ge W \iff W \le s_A$
- Per $B$: $s_B \ge W + w_A \iff W \le s_B - w_A$

Entrambe le condizioni devono valere, quindi il margine di peso consentito sopra $A$ è:
$$W \le \min(s_A,\, s_B - w_A)$$
Se scambiamo l'ordine ponendo $B$ sopra $A$, il margine diventa:
$$W \le \min(s_B,\, s_A - w_B)$$

Per far sì che l'ordine con $A$ sopra $B$ sia sempre vantaggioso (o equivalente), vogliamo:
$$\min(s_A,\, s_B - w_A) \ge \min(s_B,\, s_A - w_B)$$

Aggiungendo $(w_A + w_B)$ all'argomento del minimo:
$$\min(s_A + w_A + w_B,\, s_B + w_B) \ge \min(s_B + w_B + w_A,\, s_A + w_A)$$

Poiché $s_A + w_A + w_B > s_A + w_A$, la disuguaglianza è garantita se:
$$w_A + s_A \le w_B + s_B$$

> **Regola fondamentale di ordinamento:**
> In qualsiasi torre ottimale, le scatole possono sempre essere ordinate dall'alto verso il basso per valore crescente di $(w_i + s_i)$ (oppure dal basso verso l'alto per $(w_i + s_i)$ decrescente).

### Versione base: $n \le 2000$ (Programmazione Dinamica)
Ordinando tutte le $n$ scatole in ordine crescente di $w_i + s_i$:
$$(w_1 + s_1) \le (w_2 + s_2) \le \dots \le (w_n + s_n)$$
Ogni torre valida corrisponderà a una **sottosequenza** di questo array ordinato (dall'alto verso il basso). Possiamo definire lo stato DP in due modi equivalenti:
- $\text{dp}[k]$: il **minimo peso totale** di una torre formata da esattamente $k$ scatole.
- Inizializzazione: $\text{dp}[0] = 0$, $\text{dp}[k] = \infty$ per $k > 0$.

Iterando sulle scatole da $i = 1$ a $n$: Per ogni scatola $(w_i, s_i)$, possiamo posizionarla **sotto** una torre valida di $k$ scatole solo se il peso della torre sovrastante rispetta la sua resistenza: $\text{dp}[k] \le s_i$.

Se la condizione è verificata:
$$\text{dp}[k + 1] = \min(\text{dp}[k + 1],\, \text{dp}[k] + w_i)$$

L'aggiornamento avviene a ritroso (da $k = i$ verso $0$) come in un classico Knapsack 0/1:
```c++
// O(N^2) DP
sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
    return a.w + a.s < b.w + b.s;
});

vector<long long> dp(n + 1, INF);
dp[0] = 0;

for (const auto& box : boxes) {
    for (int k = n - 1; k >= 0; --k) {
        if (dp[k] <= box.s) {
            dp[k + 1] = min(dp[k + 1], dp[k] + box.w);
        }
    }
}

int ans = 0;
for (int k = 1; k <= n; ++k) {
    if (dp[k] < INF) ans = k;
}
```

- **Complessità temporale:** $O(n \log n + n^2) = O(n^2)$
- **Complessità spaziale:** $O(n)$

### Bonus: $n \le 10^5$ (Greedy con Priority Queue / Anti-Greedy)
Quando $n \le 10^5$, l'algoritmo $O(n^2)$ va in Time Limit Exceeded. Per risolverlo in $O(n \log n)$ si utilizza una strategia **Greedy con ritrattazione** (detta anche _Buy Low Sell High_ o _Replacement Greedy_).
### Idea dell'algoritmo
1. Ordiniamo le scatole per $(w_i + s_i)$ **crescente** (dall'alto verso il basso).
2. Man mano che scorriamo le scatole, cerchiamo di aggiungere la scatola corrente al fondo della torre corrente.
3. Se la scatola $i$ può sostenere il peso totale accumulato finora ($W_{\text{tot}} \le s_i$):
    - La accettiamo nella nostra torre.
    - Aggiungiamo il suo peso $w_i$ al peso totale $W_{\text{tot}}$.
    - Inseriamo $w_i$ in una **Max-Heap (Priority Queue)**.

4. Se invece la scatola $i$ **non** può sostenere il peso attuale ($W_{\text{tot}} > s_i$):
    - Verifichiamo se conviene fare una sostituzione: se nella torre abbiamo già preso una scatola con peso strettamente maggiore di $w_i$ (ossia $\max(w) > w_i$ al vertice della max-heap):
        - Rimuoviamo la scatola più pesante ($\max\_w$) dalla torre e inseriamo al suo posto la scatola corrente $w_i$.
        - Il numero totale di scatole nella torre rimane invariato.
        - Il peso totale diminuisce: $W_{\text{tot}} = W_{\text{tot}} - \max\_w + w_i$, lasciando più margine per le scatole successive.

##### Perché funziona la sostituzione?
Poiché stiamo scorrendo le scatole in ordine di $(w_i + s_i)$, la scatola rimossa $\text{box}_{\text{old}}$ aveva $w_{\text{old}} + s_{\text{old}} \le w_i + s_i$.
Se $w_{\text{old}} > w_i$, allora:
$$s_{\text{old}} \le (w_i - w_{\text{old}}) + s_i < s_i$$
La nuova scatola ha sia un peso minore, sia una resistenza maggiore rispetto a quella rimossa. Sostituirla rende tutti i vincoli futuri più facili da soddisfare senza violare la validità strutturale della torre.

```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>

struct Box {
    long long w, s;
};

int maxBoxesInTower(std::vector<Box>& boxes) {
    // 1. Ordina per w + s crescente
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return (a.w + a.s) < (b.w + b.s);
    });

    std::priority_queue<long long> max_heap; // Pesi delle scatole attualmente scelte
    long long current_weight = 0;

    // 2. Greedy con ripensamento
    for (const auto& box : boxes) {
        if (current_weight <= box.s) {
            // Possiamo aggiungere la scatola
            max_heap.push(box.w);
            current_weight += box.w;
        } else if (!max_heap.empty() && max_heap.top() > box.w) {
            // Sostituiamo la scatola più pesante con una più leggera
            current_weight -= max_heap.top();
            max_heap.pop();
            max_heap.push(box.w);
            current_weight += box.w;
        }
    }

    return max_heap.size();
}
```

### Confronto delle Complessità

|**Parametro**|**Approccio DP**|**Approccio Greedy + Heap**|
|---|---|---|
|**Ordinamento**|$O(n \log n)$ su $(w_i + s_i)$|$O(n \log n)$ su $(w_i + s_i)$|
|**Tempo di elaborazione**|$O(n^2)$|$O(n \log n)$|
|**Spazio aggiuntivo**|$O(n)$|$O(n)$|
|**Limite $n$ ammissibile**|Fino a $\approx 2\,000$|Fino a $\approx 10^5 - 10^6$|
# References
- https://codeforces.com/blog/entry/63533