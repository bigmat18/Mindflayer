---
Data: 2026-10-28T17:00:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Random Sapling]]"
Area:
  - "[[Master's Degree.base]]"
---
# Introduction to Random Sampling

In this part we will discuss about a problem that is simple to state, is the backbone of many randomized algorithms, and admits of solutions that are algorithmic-ally challenging to design and analyze. 

> [!NOTE] Definition
> Given a sequence of $n$ items $S=(i_1, i_2, \dots, i_n)$ and a positive integer $m \leq n$ the goal is to select a subset of $m$ items from $S$ uniformly at random.

Some important aspects in this definition:
- The term **Uniformaly** here means that any item is $S$ has to be sampled with probability $1/n$. 
- In the former scenario, the input size $n$ is known and items occupy a contiguous sequence of pages.

We will address both scenarios (known and unknwon lenght), aiming at efficiency in terms of I/Os, extra space required for the computation (in addition to the input), and amount of randomness deployed (expressed as number of randomly generated integers)

To discuss this problem we will write a `RAND(a,b)` function that randomly selects a number within the range $(a,b]$. The design of a good `RAND` will be to focus on the sampling procedure rather than on the generation of the random numbers.

### Knonw Length
#### [[Random Sampling Disk Model]]

#### [[Random Sampling Straming Model]]

### Unknown Length
#### [[Random Sampling Streaming Unknown-Length]]


# References