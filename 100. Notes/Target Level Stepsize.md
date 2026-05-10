---
Data: 2026-05-10T15:20:00
Tags:
  - note
  - youngling
Connection:
  - "[[Computational mathematics for learning and data analysis]]"
  - "[[Nonsmooth Convex Unconstrained Multivariante Optimization]]"
Area: "[[Master's degree]]"
---
# Target Level Stepsize

"If you don't know $f_*$, estimate it, but be ready to revise your estimate". Since the [[Polyak Stepsize]] requires $f_*$, we can replace it with a dynamically updated "Target Level" approximation.

![[Pasted image 20260510153507.png]]

Here is the pseudocode detailing how this logic is implemented in practice:
```c
procedure x = SGPTL(f, x, i_max, \beta, \delta_0, R, \rho)
    r = 0; 
    \delta = \delta_0; 
    f_ref = \overline{f} = f(x); 
    i = 1;

    while (i < i_max) do
        g \in \partial f(x); 
        
        // Calculate stepsize using Target Level instead of f_*
        \alpha = \beta * (f(x) - (f_ref - \delta)) / ||g||^2;
        
        // Update position
        x = x - \alpha * g; 
        
        // Check for "Good improvement"
        if (f(x) \le f_ref - \delta/2) then {
            f_ref = \overline{f}; 
            r = 0; 
        }
        // Check for "Too many steps without improvement"
        else if (r > R) then {
            \delta = \delta * \rho; 
            r = 0; 
        }
        // Accumulate distance without improvement
        else {
            r = r + \alpha * ||g||;
        }
        
        // Update record value and increment
        \overline{f} = \min\{\overline{f}, f(x)\}; 
        i = i + 1;
```

(Note: The source transcript contains a typo "f(f(x)..." which represents the `if` statement for checking the condition)

**Understanding the Pseudocode:**
- **The Target:** The algorithm uses a reference value $f_{ref}$ (the best value seen recently) and a threshold $\delta$. The target level is $f_{ref} - \delta \approx f_*$.
- **The Stepsize:** It computes $\alpha$ exactly like the Polyak stepsize, but swaps $f_*$ for the target level $(f_{ref} - \delta)$.
- **Updating Success:** If the algorithm reaches or exceeds half of the targeted gap ($f(x) \le f_{ref} - \delta/2$), the guess was good. It updates the reference value to the new lowest record ($\overline{f}$) and resets the patience counter ($r=0$).
- **Updating Failure:** The variable $r$ acts as a tracker for how much distance the algorithm has traveled without seeing a significant improvement ($r = r + \alpha ||g||$). If it travels too far ($r > R$), it assumes the target was too ambitious. It shrinks the threshold $\delta$ by a factor of $\rho \in (0,1)$ and resets the counter.

While this creates a working algorithm that converges ($\overline{f}^i \rightarrow f_*$), it introduces many ugly hyper-parameters ($\rho \in (0,1)$, $\beta \in (0,2)$, initial $\delta_0 > 0$, patience limit $R > 0$). Furthermore, it still lacks a reasonable stopping criterion other than "stop after a while".

# References