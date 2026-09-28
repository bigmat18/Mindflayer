---
Data: 2026-09-28T19:00:00
Tags:
  - note
  - padawan
Connection:
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's Degree.base]]"
---
# Boyer Moore Algorithm

This is an efficient **string-searching algorithm** that is the standard benchmark for practical string-search literature. The algorithm pre-process the **string** being searched for (the pattern), but not the string being searched in (the text).

First of all **definitions** of the terms used to descrive this algorithm:
- `T` denotes the input text to be searched. Its length is `n`.
- `P` denotes the string to be searched for, called the pattern. Its length is `m`.
- `S[i]` denotes the character at index `i` of string `S`, counting from `1`.
- `S[i..j]` denotes the substring of string `S` starting at index `i` and ending at `j`, inclusive.
- A prefix of `S` is a substring `S[1..i]` for some `i` in range `[1, l]`, where `l` is the length of `S`.
- A suffix of `S` is a substring `S[i..l]` for some `i` in range `[1, l]`, where `l` is the length of `S`.
- An alignment of `P` to `T` is an index `k` in `T` such that the last character of `P` is aligned with index `k` of `T`
- A match or occurrence of `P` occurs at an alignment `k` if `P` is equivalent to `T[(k-m+1)..k]`.

The algorithm searches for occurrences if **P** in **T** by performing explicit character comparisons at different alignments. this algorithm **use information gained by pre-processing P to skip as many alignments as possible**

The **key insight** is that if **the end of the pattern (`P`) is compered to the text**, then we can make jumps along the text rather that checking every character of the text.

The reason that this works is that lining up the pattern against the text, the last character of  the pattern is compared to the corresponding character in the text. If the character doesn't match, there is no need to continue searching backwards along the text. There are two options:
1. if the character in the text doesn't match any of the characters in the pattern, then the next character in the text to check is located $m$ characters farther along the text, where $m$ is the length of the pattern. So that means that **we know how many char in the text we can skip**
2. if the character in the text is in the pattern, then a **partial shift** of the pattern along the text is done to line up the matching character in the text decreases the number of comparisons that have to be made, which is the key to the efficiency of the algorithm.

![[Pasted image 20260928211805.png]]

More formally, the algorithm begins at alignment $k=m$, so the start of $P$ is aligned with the start of $T$.
1. Characters in $P$ and $T$ are then compared starting at index $m$ in $P$ and $k$ in $T$, moving backward
2. The strings are matched from the end of $P$ to the start of $P$
	- We continue to compare until either the beginning of P is reached (which means there is a match)
	- Or we find a mismatch, in this case we will perform a **shifted forward** (to right) according to the maximum value permitted by a number of rules.
3. The comparisons are performed again at the new alignment, and the process repeats until the alignment is shifted past the end of $T$, which means no further matches will be found.

### Shift Rule
The shift rules are implemented as constant-time table lookups, using tables generated during the preprocessing of P. The actual shifting offset **is the maximum of the shifts calculated by these rules.**
#### Bad-Character Rule
The bad-character rule considers the character in $T$ at which the comparison **process failed (assuming such a failure occurred)**. There are two kinds of mismetch

1. **The mismatch becomes a match:** The next occurrence of that character to the left in $P$ is found, and a shift which brings that occurrence in line with the mismatched occurrence in $T$ is proposed.

![[Pasted image 20260928223005.png|406]]

In the above example, we got a mismatch at position 3. Here our mismatching character is “A”. Now we will search for last occurrence of “A” in pattern. We got “A” at position 1 in pattern (displayed in Blue) and this is the last occurrence of it. Now we will shift pattern 2 times so that “A” in pattern get aligned with “A” in text.

2. **Pattern P moves past the mismatched character:** If the mismatched character does not occur to the left in $P$ a shift is proposed that moves the entirety of $P$ past the point of mismatch

![[Pasted image 20260928223058.png|418]]

Here we have a mismatch at position 7. The mismatching character “C” does not exist in pattern before position 7 so we’ll shift pattern past to the position 7 and eventually in above example we have got a perfect match of pattern (displayed in Green). We are doing this because “C” does not exist in the pattern so at every shift before position 7 we will get mismatch and our search will be fruitless.

![[Pasted image 20260928215037.png|275]]

##### Pre-Processing
We usually use a **constant-time lookup** as follows:
- Create a 2D table which is indexed first by the index of the character $c$ in the alphabet and second by the index i in the pattern
- This lookup will return the occurrence of $c$ in $P$ with the next-highest index $j < i$ or $-1$ if there is no such occurrence
- The proposed shift will then be $i-j$ with $O(1)$ lookup time and $O(km)$ space, assuming a finite alphabet of length $k$

```c++
#define NO_OF_CHARS 256

// The preprocessing function for Boyer Moore's bad character heuristic
void badCharHeuristic(string str, int size,
                      int badchar[NO_OF_CHARS])
{
    int i;

    // Initialize all occurrences as -1
    for (i = 0; i < NO_OF_CHARS; i++)
        badchar[i] = -1;

    // Fill the actual value of last occurrence of a character
    for (i = 0; i < size; i++)
        badchar[(int)str[i]] = i;
}

/* A pattern searching function that uses Bad Character Heuristic of Boyer Moore Algorithm */
void search(string txt, string pat)
{
    int m = pat.size();
    int n = txt.size();

    int badchar[NO_OF_CHARS];

    /* Fill the bad character array by calling the preprocessing function badCharHeuristic()
    for given pattern */
    badCharHeuristic(pat, m, badchar);

    int s = 0; // s is shift of the pattern with
               // respect to text
    while (s <= (n - m)) {
        int j = m - 1;

        /* Keep reducing index j of pattern while
        characters of pattern and text are
        matching at this shift s */
        while (j >= 0 && pat[j] == txt[s + j])
            j--;

        /* If the pattern is present at current
        shift, then index j will become -1 after
        the above loop */
        if (j < 0) {
            cout << "pattern occurs at shift = " << s
                 << endl;

            /* Shift the pattern so that the next
            character in text aligns with the last
            occurrence of it in pattern.
            The condition s+m < n is necessary for
            the case when pattern occurs at the end
            of text */
            s += (s + m < n) ? m - badchar[txt[s + m]] : 1;
        }

        else
            /* Shift the pattern so that the bad character in text aligns with the last
            occurrence of it in pattern. The max function is used to make sure that we get a 
            positive shift. We may get a negative shift if the last occurrence of bad 
            character in pattern is on the right side of the current character. */
            s += max(1, j - badchar[txt[s + j]]);
    }
}
```

#### Good-Suffix Rule
Like the **bad-character rule** it also exploits the algorithm's feature of comparisons beginning at the end of the pattern and proceeding towards the pattern's start. It can be described as follows:

Suppose of a given alignment of $P$ and $T$, a sub-string $t$ of $T$ matches a suffix of $P$ and suppose $t$ is the largest such sub-string for given alignment.
1. Then find, if it exists, the right-most copy _**t′**_ of _**t**_ in _**P**_ such that _**t′**_ is not a suffix of _**P**_ and the character to the left of _**t′**_ in _**P**_ differs from the character to the left of _**t**_ in _**P**_. Shift _**P**_ to the right so that substring _**t′**_ in _**P**_ aligns with substring _**t**_ in _**T**_.
2. If _**t′**_ does not exist, then shift the left end of _**P**_ to the right by the least amount (past the left end of _**t**_ in _**T**_) so that a prefix of the shifted pattern matches a suffix of _**t**_ in _**T**_. This includes cases where _**t**_ is an exact match of _**P**_.
3. If no such shift is possible, then shift _**P**_ by **m** (length of P) places to the right.

![[Pasted image 20260928215050.png]]

##### Pre-Processing
The good-suffix rule requires two tables:
1. one for use in the **general case**
2. and another for use when **the general case returns no meaningful result**

These tables will be designed $L$ and $H$ respectively. Their definitions are as follows:
1. For each $i$, $L[i]$ is the largest position less than $m$ such that string $P[i..m]$ matches a suffix of $P[1..L[i]]$ and such that the character preceding that suffix is not equal to $P[i-1]$. $L[i]$ is defined to be zero if there is no position satisfying the condition.
2. Let $H[i]$ denote the length of the largest suffix of $P[i..m]$ that is also a prefix of $P$, if one exists. If none exists, let $H[i]$ be zero.

Both of these tables are constructible in $O(m)$ **time** and use $O(m)$ space. The alignment shift for index $i$ in $P$ is given by $m-L[i]$ or $m-H[i]$. $H$ should only be used if $L[i]$ is zero or a match has been found.


# References
- https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string-search_algorithm
- https://www.geeksforgeeks.org/dsa/boyer-moore-algorithm-for-pattern-searching/