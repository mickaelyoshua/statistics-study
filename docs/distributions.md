# Normal Distribution

Bell-shaped continuous probability distribution defined by mean (μ) and standard deviation (σ).

## 68-95-99.7 Rule (Empirical Rule)
- **68%** within 1σ of mean
- **95%** within 2σ of mean
- **99.7%** within 3σ of mean

## Functions

### z_score
**Formula:** `z = (x - μ) / σ`
**Use when:** Standardizing values, comparing across distributions
**Returns:** How many standard deviations from mean

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
**Note:** Density ≠ probability (continuous distributions)

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
**Formula:** Converts to z-score, then uses standard_normal_cdf
**Use when:** Finding P(X ≤ x) for any normal distribution

```python
normal_cdf(sigma=15, mu=100, x=115)  # 0.8413 (84.13% have IQ ≤ 115)
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

## References
- [Wikipedia: Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
- [Khan Academy: Normal Distribution](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/normal-distributions-library)
