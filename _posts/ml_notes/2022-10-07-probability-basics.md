---
title: Probability basics
blog_type: ml_notes
excerpt: Refresher on probability theory.
layout: post
---

# Basics
## Random variable
Given an input space U, X is a random variable that maps every element in U to a real number $$\mathcal{R}$$, i.e., $$X:U \rightarrow V$$.

Say $$U$$ is a set of all 3 digit binary numbers viz., $$U=\{000, 001, 010, 011, 100, 101, 110, 111\}$$ and X maps element to the last digit (lsb=least significant bit). Hence, $$V= \{0, 1\}$$. The probability of $$P(X=0) = 0.5$$ and $$P(X=1) = 0.5$$. Half the numbers map to $$1$$ and other to $$0$$. Now the probability of $$P(X=0)$$ is the same as the probability of sampling a number in the input space $$U$$, that will cause $$X$$ to map to $$0$$. Here we are implicitly assuming that the probability of sampling any element in $$U$$ is the same or elements in $$U$$ are distributed uniformly.

### Independence
If X and Y are two independent random variable. Then $$P(X=x,Y=y) = P(X=x)P(Y=y)$$, that is knowing that $$X$$ takes the value $$x$$, tells you nothing about what value $$Y$$ will take and vice-versa.

Let's say X and Y are binary variables i.e., $$X:U\rightarrow \{0,1\}$$ and $$Y:U \rightarrow \{0,1\}$$. In order to formally prove $$X$$ and $$Y$$ are independent, we will have to prove that the joint $$P(X=0, Y=0) = P(X=0)P(Y=0), P(X=0,Y=1) = P(X=0)P(Y=1)...$$  for all 4 possibilities.

## Gaussian or Normal distribution
$$
\mathcal{N}(x) = \frac{1}{\sigma \sqrt{2 \pi}} \exp\left(\frac{-(x-\mu)^2}{2\sigma^2}\right)
$$

# Model metrics
## Regression metrics
### SSR - Sum of squared residuals
$$SSR = \frac{1}{n}\sum_{i=1}^n\left(y_i - \hat y_i \right)^2$$

## Linearity of expectations
Let $$ X_1, X_2, \cdots, X_n $$ be random variables defined in the same sample space, then

$$ E\left[\sum_{j=1}^n X_j\right] = \sum_{j=1}^n E[X_j]$$

Linearity expecations works even in the random variables are not independent. This
is certainly not true for products of random variables.
