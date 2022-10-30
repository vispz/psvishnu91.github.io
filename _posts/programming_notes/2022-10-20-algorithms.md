---
title: Algorithms
blog_type: programming_notes
excerpt: Alogrithms for coding interviews.
layout: post_with_toc
---

## Master method

Master method describes the time complexity of recursive divide and conquer algorithms.
Assumes that each subproblem of the recursion is of the same size. Here $$a$$ is the
number of subproblems, $$\dfrac{n}{b}$$ is the size of each subproblem and $$O(n^d)$$ is the
work done for each subproblem.

See [Stanford lectures](https://www.youtube.com/watch?v=rXiojCN9nIs&list=PLEAYkSg4uSQ37A6_NrUnTHEKp6EkAxTMa&index=20).

$$
T(n) = a \times T \left(\dfrac{n}{b}\right) + O(n^d)
$$

$$
    T(n) = 
\begin{cases}
    O(n^d \log(n)),& \text{if } a = b^d \\
    O(n^d),              & \text{if } a \lt b^d \\
    O(n^{log_{b}^a}),              & \text{if } a \gt b^d 
\end{cases}
$$

**Recursion tree method**<br/>
<a href="/assets/Images/posts/programming_notes/recursion-tree-master-method.png">
    <img src="/assets/Images/posts/programming_notes/recursion-tree-master-method.png" alt="Recursion tree master method" width="50%"/>
</a>


## Quicksort
Average run time $$O(n \log n)$$ and worst case running time $$O(n^2)$$. The running time of quicksort depends on the choice of the pivot element. For instance, if the pivot element chosen is the first element and the array is already sorted, then the running time will be $$O(n^2)$$. This is because in each step of the recursion, array is split into a left (smaller) subarray of size 0 and right subarray of size (n-i) where is the index of this pivot. The left recursion does no work, whereas in each right recursion, we recurse through $$n-1,n-2,...1$$ elements which is quadratic time.

If at every iteration we pick the **median** as the pivot element, then the array will get perfectly split into two arrays of size $$n/2$$. This will perfectly match the mergesort situation ergo running in $$\theta(n \log n)$$ time.

Even if we can choose a pivot that splits the problem into say 25-75 split, the run time will still be $$O(n \log n)$$. We can prove this with recursion tree method and the binomial theorem.

<a href="/assets/Images/posts/programming_notes/unequal-subprobs-rec-tree-method.png">
<img src="/assets/Images/posts/programming_notes/unequal-subprobs-rec-tree-method.png" alt="Recursion Tree unequal subproblems quicksort" width="50%"/>
</a>