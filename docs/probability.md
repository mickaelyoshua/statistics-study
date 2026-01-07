# Probability

Rules for calculating event probabilities and discrete distributions.

## Basic Rules

### addition_rule
**Formula:** `P(A ∪ B) = P(A) + P(B) - P(A ∩ B)`
**Use when:** Finding probability of A OR B occurring

```python
addition_rule(pa=0.5, pb=0.3, pab=0.1)  # 0.7
```

### multiplication_rule
**Formula:** `P(A ∩ B) = P(A) * P(B)` (independent) or `P(A) * P(B|A)` (dependent)
**Use when:** Finding probability of A AND B occurring

```python
multiplication_rule(0.5, 0.4)  # 0.2
```

### bayes_theorem
**Formula:** `P(A|B) = P(A) * P(B|A) / P(B)`
**Use when:** Updating probability with new evidence

```python
bayes_theorem(pa=0.01, pb=0.05, pba=0.9)  # 0.18
```

### expected_value
**Formula:** `E[X] = Σ pᵢ * xᵢ`
**Use when:** Finding average outcome over many trials

```python
expected_value([(0.5, 10), (0.5, 20)])  # 15.0
```

## Binomial Distribution

For counting successes in n independent trials with probability p.

**Use when:** Fixed n trials, binary outcomes, constant p, independent trials

### binomial_pmf
**Formula:** `P(X=k) = C(n,k) * p^k * (1-p)^(n-k)`

```python
binomial_pmf(n=10, p=0.5, k=5)  # P(5 heads in 10 flips)
```

### binomial_mean / variance / std
**Formulas:** `μ = np`, `σ² = np(1-p)`, `σ = √(np(1-p))`

```python
binomial_mean(100, 0.3)      # 30.0
binomial_variance(100, 0.3)  # 21.0
```

## Poisson Distribution

For counting events in fixed interval when events occur at constant average rate.

**Use when:** Counting rare events, known average rate, events independent

### poisson_pmf
**Formula:** `P(X=k) = (e^(-λ) * λ^k) / k!`

```python
poisson_pmf(k=5, rate=4)  # P(5 events when average is 4)
```

### poisson_cdf
**Formula:** `P(X ≤ k) = Σ P(X=i)` for i from 0 to k

```python
poisson_cdf(k=5, rate=4)      # P(at most 5 events)
1 - poisson_cdf(k=5, rate=4)  # P(more than 5 events)
```

## Central Limit Theorem

Sample means approximate normal distribution as sample size increases, regardless of population distribution.

```python
means = get_sample_means(1000, 30, coin_toss)
print_histogram(means)  # Shows bell curve
```

## References
- [Wikipedia: Probability](https://en.wikipedia.org/wiki/Probability)
- [Khan Academy: Probability](https://www.khanacademy.org/math/statistics-probability/probability-library)
- [Wikipedia: Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
- [Wikipedia: Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
