---
title: Multivariate Calculus
blog_type: ml_notes
excerpt: Notes based on ICL's brilliant course or coursera.
layout: post_with_toc
---

## Acknowledgement
Notes are primarily derived from [ICL's course on coursera](https://www.coursera.org/learn/multivariate-calculus-machine-learning/home/welcome).

[Cheatsheet](/assets/Docs/posts/ml_notes/mv-calc-cheat-sheet.pdf) from ICL.

## Gradient 
<center>
<img src="/assets/Images/posts/ml_notes/gradient-basics.png" />
</center>

$$
Grad \approx \frac{rise}{run} \approx \frac{f(x+\Delta x)-f(x)}{\Delta x}
$$

$$
f'(x)=lim_{\Delta x \rightarrow 0}\frac{f(x+\Delta x)-f(x)}{\Delta x}
$$

To compute the tangent or slope at a point, draw a straight line perpendicular to the curve. Take any points on this slope line and divide the value of the function at this point by distance between the points.
### Example

$$
f(x) = 5x^2
$$

$$
f'(x) = lim_{\Delta x \rightarrow 0} \frac{5(x+\Delta x)^2-5x^2}{\Delta x}
$$

$$
f'(x) = lim_{\Delta x \rightarrow 0} \frac{10 x \Delta x + 5 {\Delta x}^2}{\Delta x}

$$

$$
f'(x) = lim_{\Delta x \rightarrow 0} 10x + 5\Delta x
$$

$$
f'(x) = 10x
$$

### What is a variable?
Sometimes $$y = f(x)$$ makes sense but $$x = g(y)$$ does not make sense. This is because a function can have only one output. For a single input time, a car will have exactly one speed but for a single speed there could be multiple times where the car was at that speed. In other $$g(y)$$ is not a function but a relation.

## Total derivative

$$
\frac{df(x,y,z)}{dt}=\dfrac{\partial f}{\partial x}\dfrac{d x}{d t} + \dfrac{\partial f}{\partial y}\frac{d y}{d t} + \frac{\partial f}{\partial z}\frac{d z }{d t}
$$

Imagine you want to find the rate of change of temperate faced by a particle wrt to time. The temperature in the space is constant, however the particle is moving in space. So, temperature is a function of x, y, z and these are functions of t.

## Jacobian
If you have $$f(x_1, x_2, x_3, ..., x_n)$$, the Jacobian $$J$$ is given by $$J = \begin{bmatrix} \dfrac{\partial f}{\partial x_1}, \dfrac{\partial f}{\partial x_2}, \dfrac{\partial f}{\partial x_3}, ..., \dfrac{\partial f}{\partial x_n}\end{bmatrix}$$. As a convention this is written as a row vector.

Example:
$$
f = xy^2 + 2z
$$

$$
J = \begin{bmatrix}    y^2 & 2xy & 2 \end{bmatrix} 
$$

The Jacobian can be thought of as an algebraic expression for a vector, which when give a specific $$x, y, z$$ coordinate will return a vector pointing in the direction of the steepest slope for this function at $$x, y, z$$. This function has a constant contribution in the $$z$$ direction independent of the $$x,y,z$$ location chosen.

Jacobian is a vector pointing in the direction of the steepest uphill slope. The steeper the slope, greater the magnitude of the Jacobian at the point.
