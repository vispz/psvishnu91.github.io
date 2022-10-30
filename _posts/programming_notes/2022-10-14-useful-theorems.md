---
title: Useful Theorems
blog_type: programming_notes
excerpt: Theorems for coding interviews.
layout: post
---

## Fermat's little theorem

If $$p$$ is prime, for any integer $$a$$, then $$a^p-a$$ is divisible by $$p$$. If $$a$$ is divisible by $$p$$, this is trivially true.

$$ a^p \equiv a\ (\textrm{mod}\ p)$$

If $$a$$ is not divisible by $$p$$, then $$a^{p-1}-1$$ is divisible by $$p$$ as well. In other words, the remainder of $$a^{p-1}$$ when divided by $$p$$ is 1.

$$ a^{p-1} \equiv 1\ (\textrm{mod}\ p)$$

<iframe width="560" height="315" src="https://www.youtube.com/embed/w0ZQvZLx2KA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Example usage of fermat's little theorem

> **Takeaway**:
> Simplify $$a^n\ \%\ p$$ to $$a^r \ \% \ p$$, where r is $$n \ \% \ p-1$$ if p is prime.

Say we want the remainder of $$3^{100,000}$$ with $$53$$ where $$53$$ is prime.

We know that, $$a^{52}\%53 = 1$$. We can write, $$3^{100,000}=(3^{52})^{q}*3^r$$ where $$100,000 = q*52+r$$. Programmatically,

```python
q = 100_000 // 52
r = 100_000 % 52
```

We can ignore $$(3^{52})^{q}$$ as the remainder will be $$1$$ multiplied by $$1$$, $$q$$ times, ie., $$1$$. So $$3^{100,000}\ \%\ 53=3^r\ \%\ 53$$. In this example, r is 4. So the answer is $$3^4$$ % $$53$$ or $$81\ \% \ 53$$ which is 28.

Mathematically, this can be written as
$$3^{100,000}=(3^{52})^{1923} \cdot 3^4$$
$$3^{52} \equiv 1\ (\text{mod}\ 53)$$
$$(3^{52})^{1923} \equiv 1^{1923}\ (\text{mod}\ 53)$$
$$(3^{52})^{1923} \equiv 1\ (\text{mod}\ 53)$$
$$(3^{52})^{1923}\cdot 3^4 \equiv 3^4 \ (\text{mod}\ 53)$$
$$3^{100,000} \equiv 81 \ (\text{mod}\ 53)$$
$$3^{100,000} \equiv 28 \ (\text{mod}\ 53)$$

<br/>
## 2. Finding Greatest common divisor (GCD) <br/>or Highest Common Factor (HCF)
<br/>
The key here is the **Euclidean algorithm**. $$gcd(a,b) = gcd(a,b-a)$$. That is GCD of two numbers is the same if we subtract the smaller number from the bigger. We can repeatedly subtract the smaller number from the bigger number without sacrificing correctness.

Proof: Let $$p$$ be the GCD of $$a$$ and $$b$$. We can write $$a$$ and $$b$$ as $$a = p \times m, b = p \times n$$. Now, $$a-b=p\times(m-n)$$. As we can see the GCD is preserved. Repeatedly subtracting merely changes the coefs of $$m$$ and $$n$$. For instance, $$a-b-b=p\times(m-2n)$$.

<a href="/assets/Images/posts/programming_notes/euclid-theorem-proof.png"><img src="/assets/Images/posts/programming_notes/euclid-theorem-proof.png" alt="Euclid theorem formal proof" width="400"/></a>



