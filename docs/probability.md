# Probability

Rules and distributions for calculating event probabilities.

## Basic Rules

### Complement Rule
**Formula:** `P(A') = 1 - P(A)`
**Use when:** Finding probability of event NOT occurring

### Addition Rule
**Formula:** `P(A ∪ B) = P(A) + P(B) - P(A ∩ B)`
**Use when:** Finding P(A OR B)

```python
addition_rule(pa=0.5, pb=0.3, pab=0.1)  # 0.7
```

**If mutually exclusive (A ∩ B = ∅):** `P(A ∪ B) = P(A) + P(B)`

### Multiplication Rule
**Formula:** `P(A ∩ B) = P(A) × P(B)` (independent) or `P(A) × P(B|A)` (dependent)
**Use when:** Finding P(A AND B)

```python
multiplication_rule(0.5, 0.4)  # 0.2
```

**Independence:** Events are independent if `P(B|A) = P(B)`

### Bayes' Theorem
**Formula:** `P(A|B) = P(A) × P(B|A) / P(B)`
**Use when:** Updating probability with new evidence

```python
bayes_theorem(pa=0.01, pb=0.05, pba=0.9)  # 0.18
```

**Example:** Medical test with 1% disease rate, 90% sensitivity, 5% positive rate
→ P(disease|positive) = 18%

### Expected Value
**Formula:** `E[X] = Σ pᵢ × xᵢ`
**Use when:** Finding average outcome over many trials

```python
expected_value([(0.5, 10), (0.5, 20)])  # 15.0
```

## Binomial Distribution

Counting successes in n independent trials with constant probability p.

**Conditions:**
1. Fixed number of trials (n)
2. Two outcomes (success/failure)
3. Constant probability (p)
4. Independent trials

### binomial_pmf
**Formula:** `P(X=k) = C(n,k) × p^k × (1-p)^(n-k)`

```python
binomial_pmf(n=10, p=0.5, k=5)  # P(5 heads in 10 flips)
```

### binomial_mean / variance / std
**Formulas:** `μ = np`, `σ² = np(1-p)`, `σ = √(np(1-p))`

```python
binomial_mean(100, 0.3)      # 30.0
binomial_variance(100, 0.3)  # 21.0
```

**Normal Approximation:** When np ≥ 10 and n(1-p) ≥ 10, binomial ≈ normal

## Poisson Distribution

Counting events in fixed interval with constant average rate.

**Conditions:**
1. Events occur independently
2. Known average rate (λ)
3. Events don't occur simultaneously

**Use when:** Rare events, large n with small p (np = λ)

### poisson_pmf
**Formula:** `P(X=k) = (e^(-λ) × λ^k) / k!`

```python
poisson_pmf(k=5, rate=4)  # P(5 events when average is 4)
```

### poisson_cdf
**Formula:** `P(X ≤ k) = Σ P(X=i)` for i from 0 to k

```python
poisson_cdf(k=5, rate=4)      # P(at most 5 events)
1 - poisson_cdf(k=5, rate=4)  # P(more than 5 events)
```

## Binomial vs Poisson

| Binomial | Poisson |
|----------|---------|
| Fixed n trials | Events in interval |
| Known p | Known rate λ |
| Exact count | Rare events |
| np < 10 or n(1-p) < 10 | Use when np = λ, large n, small p |

## Central Limit Theorem

**Statement:** Sample means approach normal distribution as n increases, regardless of population distribution.

**Key points:**
- Works for any distribution (uniform, exponential, etc.)
- n ≥ 30 generally sufficient
- Sample means: μ_x̄ = μ, σ_x̄ = σ/√n

```python
means = get_sample_means(1000, 30, coin_toss)
print_histogram(means)  # Shows bell curve
```

**Why it matters:** Enables hypothesis testing and confidence intervals for any population.

## References
- [Wikipedia: Probability](https://en.wikipedia.org/wiki/Probability)
- [Khan Academy: Probability](https://www.khanacademy.org/math/statistics-probability/probability-library)
- [Wikipedia: Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
- [Wikipedia: Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution)
- [Wikipedia: Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)
