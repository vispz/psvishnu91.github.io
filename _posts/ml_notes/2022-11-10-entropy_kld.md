---
title: Minimising Cross Entropy = <br/> Minimising KL-Divergence?
blog_type: ml_notes
excerpt: What is entropy and KL-Divergence? Why minimising either, the same?
layout: post_with_toc
---
## Entropy & KL-Divergence

> Below $$ \log = \ln $$

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
    return logs.mean()

X: Vector = sample_from_true_distrbn_p_of_x(size=100)
print(cross_entropy(qx=q(x=X)))
```

### KL-Divergence
Read the treatise on KL-D from [UIUC](http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf).

KL Divergence is a measure of closeness of two distributions. It is not a distance metric
as it is asymmetric, more concretely, $$D_{KL}(p(x)||q(x)) \neq D_{KL}(q(x)||p(x))$$.
As mentioned above in ML, $$q(x)$$ is usually the model we are building to approximate
the true distribution $$p(x)$$.

**Discrete case**

$$D_{KL}(p(x)||q(x)) =  \sum_{x \in X} P(x) \log p(x) - P(x) \log q(x) =  \sum_{x \in X} P(x) \log \frac{p(x)}{q(x)}$$

**Continuous case**

$$D_{KL}(p(x)||q(x)) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)}$$

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
> dataset only depends on the $$H(p(x), q_{\Theta}(x))$$ term, optimising one is the
> same as optimising the other.

$$ 
\begin{align*}
D_{KL}(p(x)||q(x)) &= \sum_{x \in X} \left\{ p(x) \log p(x) - p(x) \log q(x) \right\}\\
D_{KL}(p(x)||q(x)) &= E_{x\sim p(x)}\left[\log p(x)\right] - E_{x \sim p(x)} \left[\log q(x)\right]\\
\end{align*}\\
\boxed{D_{KL}(p(x)||q(x)) = \text{Entropy}\left[p(x)\right] - H(p(x), q(x))}\\
\text{KL Divergence of $q(x)$ wrt $p(x)$} = \text{Entropy of } p(x) - \text{Cross entropy of $q(x)$ wrt to the distribution $p(x)$} 
$$

Hence, we can decompose KL-Divergence into the entropy of the true unknown distribution $$p(x)$$
plus the cross entropy of $$q(x)$$ with respect to the true distribution $$p(x)$$.
While learning a model, we try to figure out the parameters ($$\Theta$$) of our model defining $$q_{\Theta}(x)$$
which minises our loss. The parameters of the model do not affect the true probability
distribution $$p(x)$$, so 

$$ 
\begin{align*}
\arg\min_{\Theta} \left\{E_{x\sim p(x)}\left[\log p(x)\right] - E_{x\sim p(x)}\left[\log q_{\Theta}(x)\right]\right\} &= \arg\min_{\Theta} \left\{-E_{x\sim p(x)} \left[\log q_{\Theta}(x)\right] \right\}(\because p(x) \perp \Theta)\\
\arg\min_{\Theta} D_{KL}(p(x)||q_\Theta(x)) &=  \arg\min_{\Theta} H(p(x),q_\Theta(x)) 
\end{align*}
$$
