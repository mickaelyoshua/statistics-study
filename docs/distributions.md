# Normal Distribution

Bell-shaped continuous probability distribution. Most important distribution in statistics due to Central Limit Theorem.

## Parameters
- **μ (mu):** Mean - center of distribution
- **σ (sigma):** Standard deviation - spread

## Empirical Rule (68-95-99.7)

| Range | Probability |
|-------|-------------|
| μ ± 1σ | 68.27% |
| μ ± 2σ | 95.45% |
| μ ± 3σ | 99.73% |

## Functions

### z_score
**Formula:** `z = (x - μ) / σ`
**Use when:** Standardizing values, comparing across distributions
**Returns:** Number of standard deviations from mean

```python
z_score(sigma=15, mu=100, x=130)  # 2.0 (2 std above mean)
```

### get_x_from_z_score
**Formula:** `x = μ + z * σ`
**Use when:** Converting z-score back to original scale

```python
get_x_from_z_score(sigma=15, mu=100, z=2.0)  # 130.0
```

### normal_pdf
**Formula:** `f(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))`
**Use when:** Finding relative likelihood at a point

**Note:** PDF gives density, not probability. For continuous distributions, P(X = exact value) = 0.

```python
normal_pdf(sigma=1, mu=0, x=0)  # 0.3989 (peak of standard normal)
```

### standard_normal_cdf
**Formula:** `Φ(z) = P(Z ≤ z)` using error function
**Use when:** Finding cumulative probability for z-scores
**Same as:** Looking up z-table

```python
standard_normal_cdf(0)     # 0.5 (50% below mean)
standard_normal_cdf(1.96)  # 0.975 (97.5%)
```

### normal_cdf
**Formula:** Convert to z-score, then use `Φ(z)`
**Use when:** Finding P(X ≤ x) for any normal distribution

```python
normal_cdf(sigma=15, mu=100, x=115)  # 0.8413 (84.13% have IQ ≤ 115)
```

### inverse_standard_normal_cdf
**Formula:** Find z where `Φ(z) = p`
**Use when:** Finding z-score for given percentile

```python
inverse_standard_normal_cdf(0.975)  # 1.96 (97.5th percentile)
inverse_standard_normal_cdf(0.5)    # 0.0 (median)
```

### inverse_normal_cdf
**Formula:** Find x where `P(X ≤ x) = p`
**Use when:** Finding percentiles, confidence interval bounds

```python
inverse_normal_cdf(sigma=15, mu=100, p=0.5)     # 100.0 (median)
inverse_normal_cdf(sigma=15, mu=100, p=0.8413)  # ~115.0 (84th percentile)
```

## Common Calculations

**P(a < X ≤ b):**
```python
normal_cdf(σ, μ, b) - normal_cdf(σ, μ, a)
```

**P(X > x):**
```python
1 - normal_cdf(σ, μ, x)
```

**Find value at percentile p:**
```python
inverse_normal_cdf(σ, μ, p)
```

## Z-Table Reference

| z | P(Z ≤ z) |
|---|----------|
| -2.0 | 0.0228 |
| -1.0 | 0.1587 |
| 0.0 | 0.5000 |
| 1.0 | 0.8413 |
| 1.645 | 0.9500 |
| 1.96 | 0.9750 |
| 2.0 | 0.9772 |
| 2.576 | 0.9950 |

## References
- [Wikipedia: Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
- [Khan Academy: Normal Distribution](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/normal-distributions-library)
