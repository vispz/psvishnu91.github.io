---
title: Python Exotic Containers
blog_type: programming_notes
excerpt: Exotic containers in python primarily used for interviews.
layout: post_with_toc_lvl4
---

### Foreword
In production environments in Python, we almost exclusively only use the
`list`, `set` and `dict` data structures. However in coding problems, we are often
expected to the exotic data strucutre realm of

1. Balanced binary tree
2. Heap or a PriorityQueue
3. Queue/Stack
4. Tries
5. Bloom filters

AFAIK there are no standard implementations for Tries and bloom filters in Python, hence
I will merely iterate on implementing the remaining datastructures. As
a bonus, I have included
- the bisect library which provides fast binary search.
- an implementation of a disjoint union set

### `sortedcontainers` library

Python does not have a balanced binary tree implementation. If you are coming from Java
land, you can use `TreeMap => sc.SortedDict` and
`TreeSet => sc.SortedList or sc.SortedSet`.

#### Note on differences from Java
Internally, these datastructures are not stored as Balanced Binary Trees or B-Trees (more
than two children) but as a list of lists. Each sublist represents a particular level
of a B-trees. This prevents dynamic memory access.

#### Time complexities
Besides `contains` the time complexities of `SortedDict`, `SortedSet` and `SortedList`
are the same. `SortedDict`, `SortedSet` internally contain a `hashmap` and a `hashset`
respectively.

| Operation | Time Complexity |
|-----------|-----------------|
| Contains | List: $$O(\log n)$$, Set, Dict $$O(1)$$ |
| Add item | $$O(\log n)$$    |
| Pop item with key or index | $$O(\log n)$$    |
| Peek item with index | $$O(\log n)$$    |
| bisect_left, bisect_right (see [bisect section](#bisect-library))| $$O(\log n)$$|
| Get index of element in list or key in Map| $$O(\log n)$$ _same as `bisect_left`_|
| Get item with key (Map only) | $$O(1)$$    |

<br/>

``` python
"""
Example usage: we want to maintain a Map that keeps tracks of min and max value
while dynamically adding and removing elements. Everything needs to be `O(log n)`.
"""
In [1]: import sortedcontainers as sc
In [2]: sd = sc.SortedDict({'b': 2, 'a':1, 'd': 4, 'e': 5, 'c': 3})
# Fetch the min, max, 1st order and n-1th order statistic
In [3]: sd.peekitem(0), sd.peekitem(1), sd.peekitem(-2), sd.peekitem(-1)
Out[3]: (('a', 1), ('b', 2), ('d', 4), ('e', 5))

# Fetch item with key, remove item with key, remove item with index
In [4]: sd['d']
Out[4]: 4
In [5]: sd.pop('d')
Out[5]: 4
In [6]: sd
Out[6]: SortedDict({'a': 1, 'b': 2, 'c': 3, 'e': 5})
In [8]: sd.popitem(-2)
Out[8]: ('c', 3)

# Assign item
In [10]: sd['f'] = 6
In [11]: sd
Out[11]: SortedDict({'a': 1, 'b': 2, 'e': 5, 'f': 6})

# Iterate in sorted order
In [12]: list(sd.irange('b', 'z'))
Out[12]: ['b', 'e', 'f']
In [14]: sd.bisect_right('e')
Out[14]: 3

# Contains check
In [15]: 'e' in sd
Out[15]: True
```

### `heapq` library
A heap provides $$O(\log n)$$ `insertion` and `extract_min`. `extract_min` pops out the
min object. Heaps represent a binary tree but are internally stored as arrays.

> `queue.PriorityQueue` is a thread safe implementation of a heap which uses
> `heapq` underneath. Unless interview deals with concurrency, you're better off
> using the more lightweight `heapq` library.

**Second order properties:**<br/>
- $$O(n)$$ heapify, convert a list of unsorted numbers to one with heap properties.
- $$O(\log n)$$ random element deletion. This is not a guarantee, some heap
    implementations may internally store a hashmap of number to index on the array
    and hence can provide this guarantee. However, most heads do not provide this
    property and you are better off using balanced binary trees.

**Heap internals:**
The array storing the min-heap represents a binary tree where element at index, `i`
is the parent of the element at indices `2*i` and `2*i + 1`. The heap invariant is that
the parent is always smaller than both children.

``` python
In [1]: import heapq

In [2]: h = [4,5,1,9,3,5,7,0]

In [3]: heapq.heapify(h)

In [4]: h
Out[4]: [0, 3, 1, 5, 4, 5, 7, 9]

In [5]: heapq.heappop(h)
Out[5]: 0

In [6]: h
Out[6]: [1, 3, 5, 5, 4, 9, 7]

In [7]: heapq.heappush(h, -1)

In [8]: h
Out[8]: [-1, 1, 5, 3, 4, 9, 7, 5]

In [9]: heapq.heappop(h)
Out[9]: -1

In [10]: h
Out[10]: [1, 3, 5, 5, 4, 9, 7]

In [11]: heap.nsmallest(2, h)
Out[11]: [1, 3]

In [12]: heap.nlargest(2, h)
Out[12]: [9, 7]
```

### Stack / Queue
There's two ways to implement stacks and queues. One with arrays and the
other is with linkedlists. In an interview it makes no sense to implement a full blown
stack interface with arrays, so might be best to use a linkedlist via `deque` instead.
In practice, `dequeue` are a bit more heavy duty than what we need as they are
threadsafe but ¯\\\_(ツ)\_/¯

#### Stack interface
``` python
from collections import deque

stk = deque(['old_item'])
stk.append('recent_item')
recent_item = stk.pop()
```
#### Queue interface
``` python
q = deque()
q.append('first_item')
q.extend(['second_item', 'third_item'])
first_item = q.popleft()
```

### `bisect` library
Binary search through a sorted array implemented in C.

#### Bisect interface
* `bisect.bisect_left(arr, target)`: Leftmost occurrence of target, if not found the
  index immediately left of the first value greater than target.
* `bisect.bisect_right(arr, target)`: Index of first element greater than target.
In otherwords, leftmost occurrence `bisect_left`, rightmost occurrence `bisect_right + 1`.

``` python
a = [1,2,3,3,5,5,5,6]
In : bisect.bisect_left(a, 3)
Out: 2

In : a[:2]
Out: [1, 2]

In : bisect.bisect_right(a, 3)
Out: 4

In : a[4:]
Out: [5, 5, 5, 6]
```

#### Insort interface
`bisect.insort_left(arr, new_val)`, `bisect.insort_right(arr, new_val)`:
  Inserts `new_val` in `arr` inplace to maintain sorted order.
  ``` python
    In [34]: import dataclasses
    In [35]: @dataclasses.dataclass(order=True)
        ...: class Task:
        ...:     order: int
        ...:     name: str

    In [36]: t = [Task(0, 'A'), Task(0, 'B'), Task(1, 'X')]
    In [37]: sorted(t)
    Out[37]: [Task(order=0, name='A'), Task(order=0, name='B'), Task(order=1, name='X')]

    In [38]: bisect.insort_left(t, Task(order=0, name='AA'))
    In [39]: bisect.insort_right(t, Task(order=0, name='BB'))

    In [40]: t
    Out[40]:
    [Task(order=0, name='A'),
    Task(order=0, name='AA'),
    Task(order=0, name='B'),
    Task(order=0, name='BB'),
    Task(order=1, name='X')]
  ```

### Disjoint Union Set (DSU) in Python

The operations provided by the datastructure are
- Amortised O(1) Union two sets.
- Amortised O(1) Find if two sets are connected

We can add other amortised O(1) operations like finding the maximum or minimum
size amongst disjoint sets.

> The actual time complexity is $$O(\alpha(n))$$ where $$\alpha$$ is the inverse
> Ackermann function which grows extremely slowly. For all reasonable sizes of n
> this can be assumed to be constant time.

``` python
class DisjointUnionSet:

    def __init__(self, n: int) -> None:
        """Constructor
        :param n: The number of nodes in the graph. These can be connected
            later using the :meth:`.union`.
        """
        # Everyone's their own parent at the beginning
        self.parent = list(range(n))
        self.rank = [1] * n

    def find_leader(self, x: int) -> int:
        """Finds the leader of x and also moves all of x's parents directly under
        the leader of x.

        :param x: Node in the graph `0<=x<n`
        :returns: The leader of the set containing index `x`.
        """
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find_leader(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Joins sets containing x and y. If set(x) is higher rank ie., taller tree than
        set(y), y will go under x and vice-versa. If they are of the same rank or height
        then we pick one randomly but also update the rank as the height has increased.
        """
        leader_x = self.find_leader(x=x)
        leader_y = self.find_leader(x=y)
        if self.rank[leader_x] > self.rank[leader_y]:
            self.parent[leader_y] = leader_x
        elif self.rank[leader_x] < self.rank[leader_y]:
            self.parent[leader_x] = leader_y
        else:
            self.parent[leader_x] = leader_y
            self.rank[leader_y] += 1

    def is_connected(self, x: int, y: int):
        return self.find_leader(x=x) == self.find_leader(x=y)
```

#### Test
```python
dsu = DisjointUnionSet(n=10)
dsu.union(x=2,y=3)
dsu.union(x=2,y=3)
dsu.union(x=4,y=3)
dsu.union(x=4,y=9)
dsu.union(x=5,y=7)

dsu.is_connected(5,7)
Out: True

dsu.is_connected(4,3)
Out: True

dsu.is_connected(5,3)
Out: False
```
