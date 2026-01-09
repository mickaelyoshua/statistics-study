# Hypothesis Testing

Making statistical decisions about populations based on sample data.

## Core Concepts

### Hypotheses
- **H₀ (Null):** Status quo, no effect (e.g., μ = 100)
- **H₁ (Alternative):** What we want to prove (e.g., μ ≠ 100)

### Significance Level (α)
Probability of Type I error. Common values: 0.01, 0.05, 0.10

### Decision Rule
- p-value < α → Reject H₀
- p-value ≥ α → Fail to reject H₀

## Error Types

| Decision | H₀ True | H₀ False |
|----------|---------|----------|
| Reject H₀ | **Type I (α)** | Correct |
| Fail to reject | Correct | **Type II (β)** |

- **Type I (α):** False positive - rejecting true H₀
- **Type II (β):** False negative - failing to reject false H₀
- **Power = 1 - β:** Probability of correctly rejecting false H₀

## P-Value

Probability of observing result at least as extreme, assuming H₀ is true.

**Interpretation:**
- p < 0.01: Strong evidence against H₀
- p < 0.05: Moderate evidence against H₀
- p ≥ 0.05: Insufficient evidence to reject H₀

## Confidence Intervals

**Formula:** `x̄ ± z * (σ/√n)`

| Confidence | z-critical |
|------------|------------|
| 90% | 1.645 |
| 95% | 1.960 |
| 99% | 2.576 |

**Interpretation:** 95% CI means if we repeated sampling, 95% of intervals would contain the true μ.

## Functions

### standard_error
**Formula:** `SE = σ / √n`
**Use when:** Estimating variability of sample means

```python
standard_error(sigma=10, n=100)  # 1.0
```

### margin_of_error
**Formula:** `ME = z * SE`
**Use when:** Calculating CI width

```python
margin_of_error(sigma=10, n=100, confidence=0.95)  # 1.96
```

### confidence_interval
**Formula:** `x̄ ± ME`
**Use when:** Estimating population parameter range

```python
confidence_interval(x_bar=100, sigma=15, n=100, confidence=0.95)
# (97.06, 102.94)
```

### z_test_statistic
**Formula:** `z = (x̄ - μ₀) / SE`
**Use when:** Testing mean with known σ and n ≥ 30

```python
z_test_statistic(x_bar=105, mu_0=100, sigma=15, n=100)  # 3.33
```

### p_value_from_z
**Use when:** Converting z-statistic to p-value

```python
p_value_from_z(1.96, "two-sided")  # 0.05
p_value_from_z(1.96, "greater")    # 0.025
```

### is_significant
**Use when:** Making statistical decision

```python
is_significant(p_value=0.03, alpha=0.05)  # True
```

## When to Use Z-Test vs T-Test

| Condition | Test |
|-----------|------|
| σ known, n ≥ 30 | Z-test |
| σ unknown, n < 30 | T-test |

## References
- [Wikipedia: Type I and II Errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)
- [Statistics By Jim: Hypothesis Testing](https://statisticsbyjim.com/hypothesis-testing/)
- [Khan Academy: Hypothesis Testing](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample)
