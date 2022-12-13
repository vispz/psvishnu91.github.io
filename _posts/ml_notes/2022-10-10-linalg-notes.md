---
title: Linear algebra notes
blog_type: ml_notes
excerpt: Linear algebra notes based on 3Brown1Blue videos and ICL course.
layout: post_with_toc
last_modified_at: 2022-10-10
---

## Acknowledgement

These are notes taken during the following courses/videos

- [Mathematics for Machine Learning: Linear Algebra - Imperial College London, Coursera](https://www.coursera.org/learn/linear-algebra-machine-learning)
- [Essense of linear algebra - 3Blue1Brown, YT](https://youtu.be/fNk_zzaMoSs).

[Linear algebra cheatsheet from UCL.](/assets/Docs/posts/ml_notes/linearalgebra-formula.pdf)

## Miscellaneous

- Hypothesis, Converse, Inverse and Contrapositive
  - If p then q; example if figure is a rectangle then it is a square (needn't be true)
  - **If hypothesis is true, then contrapositive is true. If converse is true, then inverse is true.**
  - Hyp: p -> q; Converse: q -> p; Inverse ~p -> ~q; Contrapositive ~q -> ~p
  - _Converse_: if figure is a square then it is rectangle. _Inverse:_ if shape is not a squre, then it not a rectangle.
  - _Contrapositive_: If figure is not a square, then it is not a rectangle.

## Coordinate frame and transformation matrices

There are several interpretations of a matrix.

- A function which transforms one vector into another.
- A transformation matrix: The vectors in the original space are transformed to a new location. They are represented by the original basis vectors with $$v' = Tv$$. This is because this transformation matrix transforms our unit vectors say $$\widehat{i}$$  and $$\widehat{j}$$ to the new space. The vector $$v$$ remains the same linear combination of the basis vectors in the new space but for the new $$\widehat{i}'$$ and $$\widehat{j}'$$. So this matrix multiplication $$v' = Tv$$ essentially takes the same linear combinations but for the new basis vectors. The new basis vectors are of course represented in our coordinate frame in matrix $$T$$. So $$v'$$ is represented in our coordinate frame.
- A new coordinate frame: Here we have an original coordinate frame $$\begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix}$$ and a new coordinate frame T. Vectors in the original frame can be represented in the new frame with $$T^{-1}v$$ and vectors in the new frame can be represneted in the original frame with $$Tv$$.

## Linear transformations

In linear algebra all transformations studied are linear transformations. This means

- Origin doesn't move
- All lines remain as lines

Let v be a vector in space 2-D, $$v = 2 \widehat{i} + 3 \widehat{j}$$. Now if I transform the space and I know the new $$\widehat{i}^{\prime}$$ and $$\widehat{j}^{\prime}$$, then the transformed vector v will still be the same linear combination of the transformed bases in the new space i.e., $$v = 2 \widehat{i}^{\prime} + 3 \widehat{j}^{\prime}$$.

So in the transformed space, it's coordinates are still $$[2,3]$$ and in the original space it's $$[2\widehat{i}^{\prime}+ 3 \widehat{j}^{\prime}]$$.

All linear transformations in n-dimensions only need $$n \times n$$ numbers which can be placed in a matrix. Each column of this matrix is the new basis vectors represented in the original coordinate frame.

## What is a matrix

Matrix is a function what takes in a vector and spits out an another vector.

It is a projection of the unit basis vectors to another space what is achievable through linear transformation. The projected space can be

- stretched: square becomes a rectangle
- sheared: square becomes a parallelogram
- rotated: the base of a square which was on the x-axis (one of the unit vectors) can now be in-between axes.

The columns of a matrix represent the transformed unit basis vectors. In
2-D, $$\widehat{u_1} = \begin{bmatrix} 1 \\ 0\end{bmatrix}$$ and $$\widehat{u_2} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$.

So say we have a vector $$\begin{bmatrix} 2 \\ 3\end{bmatrix}$$ in the original space. It
essentially represents
$$2 \times \begin{bmatrix} 1 \\ 0\end{bmatrix} + 3 \times \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$
or equally $$2 \widehat{u_1} + 3 \widehat{u_2}$$.

In a projected space represented by the columns of a matrix $$A$$, this vector will be $$2 \widehat{u_1}\prime + 3 \widehat{u_2}\prime$$. The primes are the projections of $$\widehat{u_1}$$ and $$\widehat{u_2}$$ according $$A$$. In other words columns of A. what is the coefficients of the unit vectors in the projected space do not change. The grid lines in the new space will still be parallel and equidistant.

> Linear system of equations: Simultaneous equations solved with matrices.

## Determinant

Determinant of matrix is the scale by which the volume of a shape changes when applying the matrix to the shape containing the volume.

When determinant is negative, the orientation of space is inverted. In a 2-D, case we move anticlockwise to go from $$\widehat{i}$$ to $$\widehat{j}$$. If after transformation, we move clockwise then the orientation of space is inverted and the determinant must be negative. For 3-D, use the right hand rule.

## Column space, null space

For a system of equations, $$Ax = v$$, the solution is got by multiplying $$A^{-1}$$. The inverse only exists of $$det(A)\neq 0$$. If determinant is zero, we lose at least one dimension by applying the transformation $$A$$. For instance, in 3-D, the 3-D space is squished to a plane, or even a line or a single dot. These are lower rank spaces of 2,1 and 0 resply.

All the output space (all the output vectors $$v$$ of $$Ax = v$$)of your matrix is called column space. This is because columns of your matrix are the new basis vectors and the span of those is the output space. If rank is number of columns, it's a full rank matrix.

> **_NOTE:_** For full rank matrices only the $$0$$ input vector ($$x$$) can produce the $$0$$ output vector $$v$$. There's is always a unique input vector what maps to an output vector. For lower rank matrices, many input vectors can fall on the zero ouptut vector. The set of input vectors $$\{x\}$$ what fall on the origin or $$0$$ output vector $$v$$ is called the **Null Space** or the **Kernel** of the matrix.

> **The row space** is the space spanned by the row vectors of a matrix. For instance 2x3 matrix spans a 2D space. Say we have 2 linearly independent vectors, then the null space of this 2x3 matrix will be a **line what is perpendicular** to both the row vectors. This can be seen from the definition of the null space. It is the space where the input vectors are mapped to the zero vector. So the dot product of a row vector and the input vector is zero, in other words it's perpendicular. For an $$m \times n$$, matrix the row space sits in a n-dimensional space. If the rank of the matrix is r, then the null space is $$n-r$$ dimensional subspace what is perpendicular to the space r dimensional space spanned by the row vectors.

## Rectangular matrices and losing dimensions

3x2 matrix takes a 2D basis vector and maps to 3D space. The column space of this matix a plane slicing through the origin. The matrix is still full rank as the full 2D span is preserved.

2 x 3 matrix, maps a 3-D vector space on to 2D space. The 3 columns are where the 3 basis vectors land on the 2D space. This is a lower rank matrix.

## Injective, surjective and bijective matrices

![Injective](https://i.stack.imgur.com/dnFp5.png)

**Injective or 1-1:**
A function is injective if each input in the domain of the functions maps to a distinct output in the range. In other words, if $$f(x_1) = f(x_2)$$, then $$x_1 = x_2$$. Contrapositive, $$x_1 \neq x_2$$, then $$f(x_1) \neq f(x_2)$$.

**Surjective or onto:**
The entire range of the function spans the entire output space.

**Bijective:**
Both injective and bijective

### Square matrix

A square matrix is either bijective or is neither injective or surjective. If a square matrix is full rank, then every input vector in the input space maps to a unique vector in the output space (injective). Since the input is the entire n-dim input space, and every vector maps to a unique output vector in the same n-dim output space, the output has to span the entire n-dim output space (surjective).

If the matrix is not full rank, then at least one dimension will be lost. Now many input vectors have to map to a single output vector (the input space of n-dim is larger than the output space of < n-dim>) (not 1-1). And at least one of the dimensions of the output space unreachable (not onto).

### Non square matrix

Say the matrix is rectangular of shape $$m \times n$$.

If it is full rank, i.e., $$rank = min \{m,n\}$$, then the matrix is. There are 2 possibilities:

- m > n, injective: We are sending lower dimensional space to higher dimension (say 3x2 matrix, we are mapping to 2-D space to 3-D). We will span a plane slicing through the origin in this 3-D space. This will be injective but not surjective.
- m < n, surjective: We are sending higher dimensions to lower dimensions (say 2x3 matrix, we are mapping to 3-D space to 2-D, 3 basis vectors are mapped to three 2-D basis vectors). One of the dimensions is collapsed. For 3-D, we can collapse either into a plane, line or a point. Here all of the smaller output space is spanned but many input vectors will map to the same output vector. Hence, this will be surjective but not injective. Also we will have a null space of cardinality greater than 1 (0 vector).

If $$rank < min \{m, n\}$$, neither injective or surjective:

- m > n. Mapping to higher dimensions. Since rank is less than n, we can't even span the lower n-dimensional input space. Can't be injective.
- m < n: Mapping to lower dimension. Since rank is less than m, we can't even span the lower dimensional output space. Can't be surjective.

## Dot product

Why is dot product the projection? Imagine we have an arbitrary line passing through the origin in 2-D. Let the unit vector along this line be $$\widehat{u}$$. The coordinates of $$\widehat{u}$$ are $$\begin{bmatrix} u_x \\ u_y \end{bmatrix}$$. Now let's project the unit vectors $$\widehat{i}$$ onto the line with unit vector $$\widehat{u}$$. Because both are $$\widehat{i}$$ and $$\widehat{u}$$ are unit vectors, the projection of $$\widehat{i}$$ onto $$\widehat{u}$$ is the same as $$\widehat{u}$$ onto $$\widehat{i}$$ which is $$u_x$$. Similarly projection of $$\widehat{j}$$ onto $$\widehat{u}$$ is $$u_y$$. The projection of the 2-D space onto the line is given by the matrix with column vectors of where the unit vectors land, which is simply $$T = \begin{bmatrix}u_x & u_y\end{bmatrix}$$. This is clearly the unit vector transpose. Now in order to project (or transform) any vector $$v$$ in the 2-D space onto the line and represent the output in the original space we simply have to compute $$Tv$$ which is a scalar. Use these as coordinates of the unit vectors in the new space which is simply $$\widehat{u}$$.

## Scaling a vector by dot product of two vectors

Say I want to scale vector $$\bar{c}$$ by dot product of $$\bar{a}$$ and $$\bar{b}$$. All vectors are represented $$n \times 1$$ column vectors.
$$\bar{c}\prime = (\bar{a} \cdot \bar{b})_{(1\times 1)} \bar{c}_{(n \times 1)}$$

But I want to rewrite so this in terms of a matrix multplication where $$\bar{b}$$ is the vector the matrix operates on. I can rewrite this as
$$\bar{c}\prime = (\bar{c} \cdot \bar{a}^T)_{(n\times n)} \bar{b}_{(n\times1)}$$

## Projecting on to new orthonormal basis vectors

Say I have a $$\bar{r} = [1, 1]$$ in my original coordinate frame $$[O]$$ with basis vectors $$[1, 0]$$ and $$[0, 1]$$. Now there is a new set of ortho-normal basis vectors $$\widehat{n_1} = \dfrac{1}{\sqrt{2}}[1, 1]$$ and $$\widehat{n_2} = \dfrac{1}{\sqrt{2}}[-1, 1]$$. These vector values are the new coordinate system's $$[N]$$ basis vectors represented in my original frame.

I can always transform $$r$$ from $$[O]$$ to new coordinates $$[N]$$ by

$$A = \dfrac{1}{\sqrt{2}}\begin{bmatrix}1 & -1 \\ 1 & 1\end{bmatrix}$$
$$\bar{r} \text{ in } [N] = A^{-1} \bar{r}$$

where A is $$[N]$$'s basis vectors in $$[O]$$.

But since the $$[N]$$'s basis vectors are orthonormal, I can simply project $$\bar{r}$$ on to $$[N]$$ by taking the dot product of $$\bar{r}$$ with $$\widehat{n_1}$$ and $$\widehat{n_2}$$ to get the first and second coordinates respectively.

This can obviously be extended to n dimensions. Say I have a set of features in my original data space. I find a set of new interesting basis space (perhaps through PCA). If I have the coordinates of the new basis vectors in my original space and these new basis vectors are orthornormal, then I can transform my original feature vectors to the new space simply through a dot product of the original features with the new basis vectors.

Say

- $$X$$ is original feacture matrix (each column is a feature).
- $$V$$ is the matrix with the new orthonormal basis vectors as columns.

Then simply $$X^{\prime} = XV$$ will transform $$X$$ in to the new space. In other words, each row of $$X^{\prime}$$ will contain the coordinates of each data point's coordinates in the new coordinate frame.

## Composition of matrices

Say we have two transformation matrices $$T_1, T_2$$ and what are applied one after another. So any vector $$v$$ transformed by these two matrices represented in the original coordinate frame is give by $$T_2 T_1 v$$.

Let's assume we are working in 2-D. The first column of $$T_1$$ is the new $$\widehat{i}^{\prime}$$ and the second column $$\widehat{j}^{\prime}$$. The basis vectors of $$T_2$$ represented in the original coordinate frame are its columns.

$$
T_2 = \begin{bmatrix}
    i_1' & j_1' \\
    i_2' & j_2'
\end{bmatrix}
$$

$$
T_1 = \begin{bmatrix}
    i_1 & j_1 \\
    i_2 & j_2
\end{bmatrix}
$$

$$
\widehat{i}_{final} = i_1 T_2(col_1) + i_2 T_2(col_2) \\
\widehat{i}_{final} = i_1 \begin{bmatrix}    i_1' \\ i_2'\end{bmatrix} + i_2 \begin{bmatrix}    j_1' \\ j_2'\end{bmatrix}
$$

Another way to visualise this: Imagine what $$T_2$$ is the first transformation matrix and the entries in $$T_1$$ are merely vectors in the original space what need to be transformed. We can transform multiple vectors at once using a $$(2\times n)$$ matrix.

## Gram schmidt process to create orthonormal bases

Given a set of n lin indep vectors $$<v_1, v_2, ..., v_n>$$ what span the space.

$$\widehat{e_1} = \frac{v_1}{||v_1||}$$
$$u_2 = v_2 - (v_2 \cdot e_1) \widehat{e_1}$$
$$\widehat{e_2} = \frac{e_2}{||e_2||}$$
$$u_3 = u_3 - (v3\cdot \widehat{e_1}) \widehat{e_1} - (v3\cdot \widehat{e_2}) \widehat{e_2}$$
$$\widehat{e_3} = \frac{u_3}{||u_3||}$$

For you $$u_3$$, we are projecting $$v_3$$ on to the plan spanned by $$v_1$$ and $$v_2$$. The projection is the projected onto the vectors $$e_1$$ and $$e_2$$. The remaining vector is the vector perpendicular to the plane and what becomes $$u_3$$.

## Transformation in a Plane or other object

First transform into the basis referred to the reflection plane, or whichever; $$E^{−1}$$.
Then do the reflection or other transformation, in the plane of the object $$T_E$$ .
Then transform back into the original basis E. So our transformed vector $$r′ = ET_EE^{-1}r$$.

## A vector plane

A plane is defined by the normal to the plane (2 dimensional vector) $$\begin{bmatrix}a \\ b \\ c \end{bmatrix}$$, $$ax+by+cz = 0$$. The normal is not unique and can be arbitrarily scaled, so we could simplify this by taking a unit normal as convention. This is easy to see because any point on the plane $$\begin{bmatrix}p1 \\ p2 \\ 0\end{bmatrix}$$ is perpendicular to the normal and the dot product should be zero. In other words $$a \times p_1 + b \times p_2 + 0 = 0$$. what is we have effectively substituted the point in the equation of the plane and it satisfies it.

## Eigen

Eigen vectors of a matrix don't change directions when you apply the matrix on to it. In other words the coordinates are scaled together. Further the factor by which the eigen vector's magnitude is scaled by applying the matrix onto the eigen vector provides the eigen value. In mathematical notation,

$$A x = \lambda x$$

###  Eigen transformation

$$A = C D C^{-1}$$
Where columns of C are the eigen vectors and the D is a diagonal matrix with the corresponding eigen vectors. Applying the transform repeatedly is merely applying the diagonal matrix D raised to the appropriate power to the vector moved into the eigen basis space.

$$A^n = C D^n C^{-1}$$

Equivalently, if you are given a set of linearly independent eigen vectors what can span the full space you can compute the eigen values with
$$D = C^{-1} A C$$

###  When can you have diagonalisation?

You may not always be able to diagonalise the matrix. Generally speaking for a n dimensional matrix, if you do not have n linearly independant eigen vectors (a transformation such as rotation by 90 degrees has 0 real eigen values and 0 eigen vectors), you cannot construct the diagonal matrix.

###  Power method

If a vector is an eigen vector of a transformation matrix, it does not rotate or change direction. All other vectors rotate when a matrix is applied. A non eigen vector is always rotated a certain amount towards each eigen vector when a matrix is applied. The proportion to which it's rotated towards each of the eigen vector is proportional to the eigen value corresponding to each eigen vector. You can imagine if you keep applying the matrix on to the vector progressively moves closer to the eigen vector with the largest eigen value, eventually converging to it.

This process can be used to extract the eigen vector with the highest eigen value. Begin with a random vector and repeatedly multiply it with the transformation matrix A until convergence. This is called the Power method.

> Notice what if you have a valid set of eigen vectors (enough to span the space), then deriving the eigen values is simple. ($$D = C^{-1} A C$$, where C is a matrix with the eigen vectors as columns).

## Shear transformation matrix in 2-D

In 2-D, off diagonal elements cause shear, if the other diagonal element is 0. This can be understood geometrically, imagine rotating the y axis but not the x-axis, a unit square in the original space will be squished or sheared.

## Abstract vector space

A function can be thought of as a infinite dimensional vector where each coordinate is each value in the input space of the function. For a simple scalar function $$f(x)$$ it's merely the number line or set of real numbers $$R$$. For a vector function, we can simply construct a index function in $$R$$ over the space of the inputs. You can always add functions with $$(f+g)(x) = f(x) + g(x)$$ or scale functions $$(2f)(x) = 2 f(x)$$.

When you do a transformation of a function, the transformation takes in a function and returns another function operating on the space. For instace derivate d/dx(x^+1) = 2x. Here d/dx is the transformation also called an operator. When can this operator or transformation be linear in regards to functions. It needs to satisfy two properties:

1. Additive: $$L(\bar{v} + \bar{w}) = L(\bar{v}) + L(\bar{w})$$
2. Scaling: $$L(c\bar{v}) = cL(\bar{v})$$

The above two are the same two conditions or operations on a vector. For instance if we transform two vectors and add them, we get the same result as transforming each vector and then adding them. Also transforming a vector and the scaling by a constant, should give the same result as scaling the vector by a constant and then transforming them.

Also it's immediately obvious what the above two conditions are true for the differentiation operator.

> **Linear transformations** preserve the operation of addition and scalar multiplication.

Linear transformations or operators can be represented by matrix multiplication, with only caveat being it's an $$(\infty \times \infty)$$ dimensional matrix.

###  Vector space

A vector space is any space where which satisfy a set of axioms (derived naturally from the additive and scalar multiplication) property. Then all the results of linear algebra can directly be applied to this vector space. The vector space, could be the set of 2D vectors or functions etc.

> **Vector field:** Appears in multivariate calculus to explain functions which map n-dimension vectors to n-dimensional vectors. Say we are dealing with 2D functions, the input will be 2-D and output will 2-D, so we need a 4-D space to visualise. Instead we place the output vectors at grid point in the input vector space. Also, we often normalise the length of the output vectors to make it easy to visualise.
