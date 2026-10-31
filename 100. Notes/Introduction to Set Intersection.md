---
Data: 2026-10-30T12:57:00
Tags:
  - note
  - youngling
Connection:
  - "[[Algorithm Engineering]]"
  - "[[Set Intersection]]"
Area:
  - "[[Master's Degree.base]]"
---
# Introduction to Set Intersection

The **Set Intersection** problem, is a basic problem in the Search Engines. A  search engine is a well-known tool designed to search for information in a collection D of documents. We analyse the specific case we just textual documents we will have:
- A document $d_i \in D$ (that can be a book, news item or any file containing text), where the search engine builds an **index** to answer efficiently to queries.
- A query $Q$ is commonly structured as a **bag of words $w_1w_2\dots w_k$**

The goal is to retrieve efficiently the **most relevant document in $D$** wiich contain alla the query words. In a more formal way, we can say:

> [!NOTE] Problem
> Given a sequence of words $Q=w_1w_2\dots w_k$ and a document collection $D$, fin the document $d\in D$ that contains all words $w_i$

Modern search engines to optimize this operation build a very simple data structure called **Inverted Index**, it is composed by three essential parts:
- **the dictionrary of words** $w$ 
- one **list of occurrences** per dictionary word (called **posting list**, which we will designate $L[w]$)
- some **additional information** indicating the importance of each of these occurrences 

> [!NOTE] Note
> The term **inverted** refers to the fact that word occurrences are not sorted according to their position in the document, but according to the alphabetic ordering of the words to which they refer.

So inverted indexed are similar to the classic glossary contained in some books, here extended to represent occurrences of all the words present in a  collection of documents.

Some important things to keep in mind:
- Each posting list $L[w]$ is stored contiguously in a single array
- The names of indexed documents are palced in another table, where they are identified through unique positive integers known as `docIDs`, these IDs are generally assigned in an arbitrary way by the search engine
- he dictionary is stored in a table which contains some satellite information plus the pointers to the posting list

![[Pasted image 20261030135919.png]]

Now let us assume that we have a query $Q = \{\text{abaco, mathematics}\}$. We need to find the documents in $D$ that contain both of the words in $Q$ boild down to finding the `docIDs` shared by the two inverted list $L[abaco]$ and $L[mathematics]$. This problem is formally called **intersection problem**.

To avoid a brute-force approach the key idea here is to sort the posting lists once for all. We thus reformulate the intersection problem on two sorted sets $A=L[abaco]$ and $B=L[mathematics]$ as:

> [!note] Problem
> **Sorted set intercetion problem** is: given two sorted integer sequences $A=a_1a_2, \dots a_n$ and $B=b_1b_2\dots b_m$ such that $a_i < a_{i+1}$ and $b_i < b_{b+1}$ compute the integers common to both sets

### [[Merge-Based Approach]]
### [[Mutual Partitioning]]
### [[Doubling Search]]
### [[Two-Level Storage Approach]]
# References