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

### Decomposition principle

To get time complexity of randomised algorithms
1. Identify a random variable Y you care about
2. Express Y as a sum of indicator random variables
$$ Y = \sum_{e=0}^m  X_e $$

3. Apply linearity of expectation: $$ E[Y] = \sum_{e=0}^m P[X_e = 1] $$

## Selection problem

Given an array A with n distinct numbers and k, return the kth order statistic. The
k-th order statistic is the kth smallest number.

10, 8, 2, 4 -> 3rd order statistic is 8. First order statistic is 2.

The solution is O(n). Using reduction, we can sort the array in O(n log n) by sorting
and returning the (k+1)th element.

#### Randomised
Follows quicksort almost verbatim.

{% include button.html url="https://github.com/psvishnu91/interview-problems/blob/master/design_of_algos/chapter_5_quicksort/selection_randomised.py" name="Github Solution" %}

**Algorithm:**
- Randomly choose a pivot and apply the partition subroutine from quicksort
- If the index of the pivot is k, return the value at the pivot.
- If the index of the pivot is greater than k (kth order statistic), then recurse on
the left half of the array, otherwise recurse on the right half of the array.

**Time complexity**
This is a linear time algorithm on expectation. The proof involves using
- Number of phases. A phase is where the input array is reduced to at least (3/4)th
the input size. Phase j=0, we are processing an array of size btw, $$\left(\frac{3}{4}\right)^1 n$$
and $$\left(\frac{3}{4}\right)^0 n$$. Phase two (j=1), an array of size btw $$\left(\frac{3}{4}\right)^2 n$$
and $$\left(\frac{3}{4}\right)^1 n$$.
- Expected number of recursions to leave a phase (2). Because 50% of numbers result in a
25/75 split, it's the same as the expected number of tosses of a coin before we see a head
$$ E[n] = 1 + 0.5E[n]$$. This is because we need at least one toss. With 0.5 probability
we are back to square one.
- Then total work becomes upper bounded by $$2 \times c \times n \sum_j \left( \frac{3}{4}\right)^j
$$. The infinite geometric sum is $$\frac{1}{1-r} = \frac{1}{1-\frac{3}{4}} = 4$$. The total
work becomes upper bounded by $$8cn$$.


### Deterministic - Median of medians

The algorithm is precisely the same as the randomised approach except how we choose our
pivots. This algorithm is deterministically guaranteed to be of $$O(n)$$. However,
we pay in terms of very large constants in the big-O notation, need for an extra storage
of length $$O\left(\frac{n}{5}\right)$$.

**Algorithm**
- First split the array into n/5 buckets of sizes 5 each. For each bucket find the median through sorting.
- Store these medians in a new list.
- Then find the median of these medians by recursively calling this function.
- Once we have the pivot the rest is the same as the randomised selection.

**Time complexity**
- We need to notice that after each round, the pivot is guaraneed to shrink the array
by at least 30% each time. Say we have 100 elements in the array. We split into 20 buckets
and arrive at 20 medians. Then we find the median of these medians (M). This will point to say
the 11th bucket. This means, there are 10 medians smaller than this (by definition). For
each bucket these 10 medians belong to there are 3 elements (including the bucket median)
smaller than the median of medians (M). That is there are 30 elements out of the 100
smaller than the  median of medians (M).
- Next we write out the recurrence as $$ T(n) = cn + T\left(\frac{n}{5}\right) + T\left(\frac{7n}{10}\right)$$.
We can then prove this with induction or a recursion tree approach.

## Graph representation

Elements: Nodes or vertices $$(V)$$ and edges $$(E)$$.

**Types**
{% include image.html id="/assets/Images/posts/programming_notes/graphs.png" width="20%" %}

Edges are directed or undirected
1. Undirected graph - The nodes are unordered.
2. Directed graph - Has directed edges. There is a first (tail) and the last vertex (head)
   or end point. Directed edge is also called an Arc. If a -> b, then a is the tail
   and b is the head.

### Terminologies
- Parallel edges: They are edges connecting the same two nodes.
- Connected graph: no nodes with zero edges.
- Complete graph: A graph with all nodes connected to each other.
- Degree of a node: Number of egdes incident on the node.


### Facts
- **Number of edges:** n-num vertices and m-num edges. For an connected undirected graph
and with no parallel edges, the min and max number of possible edges is $$(n-1)$$ and
$$\frac{n(n-1)}{2}$$.
- Sparse graph vs Dense graph: m is $$\Omega(n)$$ and $$O(n^2)$$. If m is closer to
  $$\Omega(n)$$ then it's a sparse graph, if it's closer to $$\Omega(n)$$ it's a dense graph.
- $$\sum_v degree(v) = 2m$$. Each edge is a part of two nodes.

### Adjacency matrix
For a graph with n-vertices, the adjacency matrix is a $$n\times n$$
matrix, where each entry A[i,j] represents if there is an edge btw the two nodes i and j.

**Extensions**
1. Parallel edges: A[i,j] has number of edges.
2. Weighted edges: A[i,j] has weights.
3. Directed edges: A[i,j], 1 if from i to j and -1 if from j to i.

**Space complexity:** O(n^2). Can beat this with sparse matrix representations.

### Adjacency list
We store the following datastructure

#### Programming interview style

1. Nodes are all integers from `0` to `n-1`, we use a `list[set[int]]`.  Each item in the
   list is a node and each element in the set are the nodes at the head of the outward
   arc from this node. If undirected it's simply the neighbour of this node.
2. Node are node datatype. We use `dict[Node,set[Node]]`. It's similar to the above
   and of course we need to make the node hashable.

{: .code title="Building a graph in Python" .x}
```python
def _build_graph(edges: list[list[int]], num_nodes: int) -> list[set]:
    """Builds a adjacency-list type graph from a list of edges.
    :param edges: Each item is an edge [i, j] such that we have an edge i -> j.

    Sample input::
        edges=[[1,0], [2,0], [3,2], [1,3]], num_nodes=5
        [1] -> [0]   [4]
         |      ^
         v      |
        [3] -> [2]

    Sample output::
        graph=[set(), {0,3}, {0}, {2}, set()]
    """
    adj_set = [set() for _ in range(num_nodes)]
    for e in edges:
        adj_set[e[0]].add(e[1])
    print(f"{adj_set=}")
    return adj_set
```

#### Production style

{: .code title="Undirected graph code in Python" .x}
``` python
@dataclass
class Edge:
    nodes: tuple['Node', 'Node']

@dataclass
class Node:
    val: int = 0
    edges: Optional[set['Edge']] = None

"""
Graph
        (0) -- (1)
         |
        (2)
"""
nodes = [Node(val=0), Node(val=1), Node(val=2)]
edges = [Edge(nodes=(nodes[0], nodes[1])), Edge(nodes=(node[0], node[2]))]
nodes[0].edges = [edge[0], edge[1]]
nodes[1].edges = [edge[0]]
nodes[2].edges = [edge[1]]
```

**Directed**

{: .code title="Directed graph in Python" .x}
``` python
@dataclass
class Edge:
    tail: 'Node'
    head: 'Node'

@dataclass
class Node:
    val: int = 0
    outward_edges: Optional[set['Edge']] = None
    # This variable is optional and adds to storage. We
    # can get away with just the outward edges.
    inward_edges: Optional[set['Edge']] = None

"""
Graph
        (0) --> (1)
         Ʌ
         |
        (2)
"""
nodes = [Node(val=0), Node(val=1), Node[val=2]]
edges = [
    Edge(tail=nodes[0], head=nodes[1]),
    Edge(tail=nodes[2], head=nodes[0]),
]
nodes[0].outward_edges = {edges[0]}
nodes[0].inward_edges = {edges[1]}
nodes[1].outward_edges = {}
nodes[1].inward_edges = {edges[0]}
nodes[2].outward_edges = {edges[1]}
nodes[2].inward_edges = {edges[0]}
```

**Space complexity:** $$\theta(m+n)$$. This can be $$O(n^2)$$ for a compelte graph.
However, this is an overestimate. The above complexity handles this as $$m$$ will be
$$n^2$$, in this case.

##### Which one should you choose
Depends on graph density and operations we want to support.

- For low density, high number of nodes in a graph, it's more efficient to choose
adjacency list. Imagine the internet where the number of webpages is extremely large
and the graph is rather sparse.
- For graph search, adjacency list provides the appropriate operations we need.

## Graph Algorithms
### Topological sort
Algorithm that given a directed _acyclic_ graph, returns an ordering of the nodes with
the following property:

For any two nodes in the output array `TS` with indices `i` and `j` such `i < j`, then
all outward arcs from `TS[i]` will end in a node `TS[k]` such that `k <= j`.

The only condition for a Topological Sort to exist is that the directed graph needs to
be acyclic.

#### Algorithm
The algorithm piggybacks on DFS.

- Wrapper function and a DFS helper function.
- In wrapper function, create
    * create an output list with the same length as number of nodes (n).
    * create a seen hashset.
    * Initialise a global index to n-1.
    * Iterate over all the nodes to handle disjoint nodes.
- In the dfs helper function
    * mark the node as seen
    * recurse over adjacent nodes.
    * At the end of the recursion, add the node to the current index and decrement
        the index.

{: .code title="Topological sort in Python" .x}
```python
import typing
from dataclasses import dataclass
# Graph represented as a list of adj nodes, `len(graph)` is num nodes and
# `len(graph[i])` is number of adjacent numbers from node `i`.
Graph = typing.NewType('Graph', list[list[int]])

@dataclass
class Index:
    ix: int

def topsort(graph: Graph) -> list[int]:
    if not graph:
        return []
    n = len(graph)
    output, output_ix, seen = [None]*n,  Index(ix=n-1), set()
    for i in range(n):
        if i in seen:
            continue
        _dfs(graph=graph, node=i, seen=seen, output=output, output_ix=output_ix)
    return output

def _dfs(
    graph: Graph, node: int, seen: set[int], output: list[int], output_ix: Index,
) -> None:
    seen.add(node)
    # This will be only be empty for the sink node; equally this loop will be a no-op
    # if all the descendents have already been visitied
    for adj in graph[node]:
        if adj in seen:
            continue
        _dfs(graph=graph, node=adj, seen=seen, output=output, output_ix=output_ix)
    output[output_ix.ix] = node
    output_ix.ix -= 1

#          ┌────┐
#   ┌──────► 1  ├───────┐
#   │      └────┘       │
# ┌─┴──┐             ┌──▼─┐
# │ 0  │             │ 3  │
# └─┬──┘             └──▲─┘
#   │      ┌────┐       │
#   └──────► 2  ├───────┘
#          └────┘
print(topsort(graph=Graph([[1,2],[3], [3], []])))
```
### Cycles in graphs
#### Cycles in Directed Graphs
A directed graph with no cycles is called a DAG or a Directed Acyclic graph. The
algorithm uses DFS and 3 sets, `explore`, `visiting` and `done`.

- We add all the nodes to the `explore` set.
- In the outer wrapper method, iterate over


### Graph min cuts

The goal is to split the graph of n-vertices into two non-empty sets A, B such that we have
the least number of crossing edges. Min cut problem allows for _parallel edges_.

{% include image.html id="/assets/Images/posts/programming_notes/graph-cut.png" width="80%" %}

For undirected graphs these are any edges from vertices in set A to B or vice-versa.
For directed graphs, we only count edges from A to B, i.e., tail in A and head in B.

For a graph with n-nodes, we have have $$2^n-2$$ different cuts. This is because each
node is a binary variable with two options, node 1 or node 2. We have a minus two because
the empty set cases are disallowed (all are in set A or all are in set B).
