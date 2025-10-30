---
Data: 2025-10-24T17:15:00
Tags:
  - note
  - youngling
Connection:
  - "[[Programming & Algorithms]]"
  - "[[Dynamic Programming]]"
  - "[[Competitive Programming and Contests]]"
Area: "[[Master's degree]]"
---
# Longest Palindromic Substring
Given a string **s**, find the longest substring which is a palindrome. If there are multiple answers, then find the first appearing substring.
######  Examples 

> **Input:** s = "forgeeksskeegfor"  
> **Output:** "geeksskeeg"  
> **Explanation:** The longest substring that reads the same forward and backward is "geeksskeeg". Other palindromes like "kssk" or "eeksskee" are shorter.
> 
> **Input:** s = "Geeks"  
> **Output:** "ee"  
> **Explanation:** The substring "ee" is the longest palindromic part in "Geeks". All others are shorter single characters.
> 
> **Input:** s = "abc"  
> **Output:** "a"  
> **Explanation:** No multi-letter palindromes exist. So the first character "a" is returned as the longest palindromic substring.

### Using Dynamic Programming
In this approach the use the idea to store sthe status of smaller substrigs and use these results to check if a longer substring forms a palindrome.
- The idea is that if we know the status of the substring ranging `[i,j]` we can find the status of the substring ragging `[i-1,j+1]`
- If the substring from `i` to `j` is not a palindrome, then the substring from `i-1` `j+1` is not a palindrome. Otherwise, it will be a palindrome only if `str[i-1]` and `str[j+1]` are the same.

We create a 2D table which stores status of substring `str[i...j]`. We simply store true or false if the substring is palyndrome or not
###### Example
![[Pasted image 20251024185943.png | 250]]

The longest length for which a palindrome formed will be the required answer.

```c++
string getLongestPal(string s) {
    int n = s.size();
    vector<vector<bool>> dp(n, vector<bool>(n, false));
    
    // dp[i][j] if the substring  from [i to j]
    // is a palindrome or not
    
    int start = 0, maxLen = 1;
    
    // all substrings of length 1 are palindromes
    for (int i = 0; i < n; ++i) dp[i][i] = true;
    
    // check for substrings of length 2
    for (int i = 0; i < n - 1; ++i) {
        if (s[i] == s[i+1]) {
            dp[i][i+1] = true;
              if(maxLen==1){
                    start = i;
                    maxLen = 2;
                }
        }
    }
    
    // check for substrings of length 3 and more
    for (int len = 3; len <= n; ++len) {
        for (int i = 0; i <= n - len; ++i) {
            int j = i + len - 1;
            
            // if s[i] == s[j] then check for 
            //  i [i+1  --- j-1] j 
            if (s[i] == s[j] && dp[i+1][j-1]) { 
                dp[i][j] = true;
                if(len>maxLen){
                    start = i;
                    maxLen = len;
                }
            }
        }
    }
    return s.substr(start, maxLen);
}
```

- **Time complexity** is $O(n²)$ because we need to compute half of the stored matrix $n² / 2 \approx n²$ 
- **Memory complexity** is the memory needed to store the matrix that is $O(n²)$

### Using Expansion from Center
In this version the idea is to traverse each character in the string and treat is as potential center of a palindrome, trying to expand around it in both directions while checking if the expanded substring remains a palindrome. We need to check two case for a center 
1. Where the current character is the center (**odd-length**)
2. Where the current character and the next character together from the center (**eve-length**)

For each expantion we keep track of the length and we save the best with the start value to recustrust the final string.

```c++
string getLongestPal(string &s) {
    
    int n = s.length();
    int start = 0, maxLen = 1;

    for (int i = 0; i < n; i++) {

        // this runs two times for both odd and even 
        // length palindromes. 
        // j = 0 means odd and j = 1 means even length
        for (int j = 0; j <= 1; j++) {
            int low = i;
            int high = i + j; 

            // expand substring while it is a palindrome
            // and in bounds
            while (low >= 0 && high < n && s[low] == s[high]) 
            {
                int currLen = high - low + 1;
                if (currLen > maxLen) {
                    start = low;
                    maxLen = currLen;
                }
                low--;
                high++;
            }
        }
    }

    return s.substr(start, maxLen);
}
```

- **Time complexity** is $O(n²)$ in the bad case because for each position if we have a string with all same character we iterate n * n;
- **Memory complexity** is constant because we don't need to store anything $O(1)$.

### Manacher’s Algorithm
This allow to compute in $O(n)$ time complexity. https://www.geeksforgeeks.org/dsa/manachers-algorithm-linear-time-longest-palindromic-substring-part-1/
# References
- [GeeksForGeeks Article](https://www.geeksforgeeks.org/dsa/longest-palindromic-substring/)