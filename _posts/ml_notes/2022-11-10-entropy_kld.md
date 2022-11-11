---
title: Minimising Cross Entropy = <br/> Minimising KL-Divergence?
blog_type: ml_notes
excerpt: What is entropy and KL-Divergence? Why minimising either, the same?
layout: post
---
## Entropy & KL-Divergence

### Entropy
Measures the uncertainty in a system. In an information theoretic view, it's the
amount of information needed to remove the uncertainty in the system.

$$Entropy(x) = S\left[P(x)\right] = - \sum_{x \in X}P(x)\log P(x) = - E_{x \sim P(x)} [\log P(x)]$$

Above we are sampling $$x$$ from the distribution $$P(x)$$: that is the probability of
sampling the specific value $$x$$ is defined by the probability distribution $$P(x)$$. 
> To confirm (?): My understanding of why the expectation of the $$\log P(x)$$ is entropy or uncertainty
> is because we need to $$\log_2 n$$ bits to represent n numbers
> (3-bit => 8nums, 4-bit 16 nums). Since entropy is the amount of information needed to
> remove the uncertainty, we need log of the unknown number of codes (bits) to remove
> the uncertainty.

Here's a [jupyter notebook](https://nbviewer.org/gist/psvishnu91/738cbc59e9f80fa72c3942e9aa2cfd48) 
with visualisation of entropy as a function of how spread out a distribution is.

<iframe 
    src="https://nbviewer.org/gist/psvishnu91/738cbc59e9f80fa72c3942e9aa2cfd48"
    title="Entropy as a function of how spread out a distribution is."
    width="100%" height="350"
>
</iframe>

### Cross entropy

This is the entropy of a distribution $$q(x)$$ when $$x$$ is sampled from another distribution
$$p(x)$$. 

> Paraphrasing **Wikipedia:** Cross entropy measures the average number of bits needed to represent
> a specific $$x$$ if the coding scheme used is optimized for an
> estimated probability distribution $$q(x)$$, rather than the true distribution $$p(x)$$.

$$ H(p(x), q(x)) = - E_{x \sim p(x)}\left[ \log q(x) \right] $$

In ML, $$p(x)$$ represents the true unknown distribution we are trying to model with $$q(x)$$.
We never really have access to the true distribution $$p(x)$$, but only $$X$$, our
dataset sampled from $$p(x)$$. Here $$X$$ represents the entire dataset, $$X$$ 
and $$y$$ in supervised learning, not just the features. When we actually go about 
computing cross entropy, we merely compute the average $$\log q(x)$$ with $$x$$'s 
sampled from our data. This is an unbiased estimate of the true cross entropy of
$$q(x)$$ wrt $$p(x)$$.

``` python
import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float64]

# Penalty for the model q(x) for estimating 0 probability of seeing a sample x
# from p(x).
LARGE_PENALTY = 100

def q(x: Vector) -> Vector:
    model = fetch_model()
    return model.probability_of_x(x)

def cross_entropy(qx: Vector) -> float:
    logs = np.log(qx)
    logs[logs==-np.inf] = LARGE_PENALTY
    return -logs.mean()

X: Vector = sample_from_true_distrbn_p_of_x(size=100)
print(cross_entropy(qx=q(x=X)))
```

### KL-Divergence
Read the treatise on KL-D from [UIUC](http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf).

KL Divergence is a measure of closeness of two distributions. It is not a distance metric
as it is asymmetric, more concretely, $$D_{KL}(p(x)||q(x)) \neq D_{KL}(q(x)||p(x))$$.
As mentioned above, in ML $$q(x)$$ is usually the model we are building to approximate
the true distribution $$p(x)$$.

**Discrete case**

$$D_{KL}(p(x)||q(x)) =  \sum_{x \in X} \left\{ P(x) \log p(x) - P(x) \log q(x) \right\} =  \sum_{x \in X} P(x) \log \frac{p(x)}{q(x)}$$

**Continuous case**

$$D_{KL}(p(x)||q(x)) = \int_{-\infty}^{\infty} { p(x) \log \frac{p(x)}{q(x)} dx}$$

**When $$p(x) \neq 0$$ but $$q(x) = 0$$**

If $$p(x) \neq 0$$ for some $$x \in X$$, but $$q(x) = 0$$, 
then $$D_{KL} = p(x) (\log p(x) - \log 0) = \infty$$. If there's a point which has non-zero
probability in the true distribution that the model distribution thinks is impossible,
then the two distributions are considered absolutely different. This can be addressed
by smoothing. See the [UIUC](http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf) notes
for more details.

### Why is minimising Cross-Entropy or KL-Div, the same?

Credits: [stats.stackexchange](https://stats.stackexchange.com/a/357974/84357)

> **tl;dr:** KL-Divergence is composed of entropy of the true distribution
>  $$p(x), E\left[\log p(x)\right]$$ and
> the cross entropy, $$H(p(x), q_{\Theta}(x))$$. Since optimising the
> parameters of the model $$\left(\Theta\right)$$, only depends on $$q_\Theta{(x)}$$ and 
> hence only the
> $$H(p(x), q_{\Theta}(x))$$ term, we can leave out entropy of $$p(x)$$ from the objective function.

Cross-entropy is the loss function used in logistic regression. We pair softmax and cross-entropy
loss in 

$$ 
\begin{align*}
D_{KL}(p(x)||q(x)) &= \sum_{x \in X} \left\{ p(x) \log p(x) - p(x) \log q(x) \right\} \\
 D_{KL}(p(x)||q(x)) &= E_{x\sim p(x)}\left[\log p(x)\right] - E_{x \sim p(x)} \left[\log q(x)\right]\\
\end{align*}\\
\boxed{D_{KL}(p(x)||q(x)) = -\text{Entropy}\left[p(x)\right] + H(p(x), q(x))}\\
\text{KL Divergence of $q(x)$ wrt $p(x)$} = -\text{Entropy of } p(x) + \text{Cross entropy of $q(x)$ wrt to the distribution $p(x)$} 
$$

The signs are flipped in the final equation because entropy and cross entropy are
the negative expectation of log the distributions.

Hence, we can decompose KL-Divergence into the entropy of the true unknown distribution $$p(x)$$
plus the cross entropy of $$q(x)$$ with respect to the true distribution $$p(x)$$.
While learning a model, we try to figure out the parameters $$\left(\Theta\right)$$ of our model defining $$q_{\Theta}(x)$$
which minises our loss. The parameters of the model do not affect the true probability
distribution $$p(x)$$, so 

$$ 
\begin{align*}
\arg\min_{\Theta} \left\{E_{x\sim p(x)}\left[\log p(x)\right] - E_{x\sim p(x)}\left[\log q_{\Theta}(x)\right]\right\} &= \arg\min_{\Theta} \left\{-E_{x\sim p(x)} \left[\log q_{\Theta}(x)\right] \right\}(\because p(x) \perp \Theta)\\
\arg\min_{\Theta} D_{KL}(p(x)||q_\Theta(x)) &=  \arg\min_{\Theta} H(p(x),q_\Theta(x)) 
\end{align*}
$$

## So what should you measure in production?
Well honestly, you can track either KL-Divergence $$D_{KL}$$ or Cross entropy $$H$$ in production. 
Minimising either loss is identical as we have seen above.

However, KL-Divergence is slightly easier to reason with. Let's say we have a perfect model
such that $$q(x) = p(x) \forall x \in X$$. Say that, for $$x=1$$, $$p(x)=0.5$$ and hence $$q(x)=0.5$$,
also that for $$x=2$$,  $$p(x)=q(x)=0.2$$. We would expect the cross-entropy even if not zero
should be identical 


``` python
In [1]: import numpy as np

In [2]: p_of_x = np.arange(0.1,1,0.1)

In [3]: cross_entropy = np.round(- p_of_x * np.log(p_of_x), 2)

In [4]: entropy_p_of_x = np.round(- p_of_x * np.log(p_of_x), 2)

In [5]: entropy_p_of_x
Out[5]: array([0.23, 0.32, 0.36, 0.37, 0.35, 0.31, 0.25, 0.18, 0.09])

In [6]: # if q(x) is identical to p(x), then cross entropy will be the same as entropy of p(x)
```
