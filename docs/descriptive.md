# Descriptive Statistics

Summarize and describe data characteristics.

## Central Tendencies

### mean
**Formula:** `x̄ = Σxᵢ / n`
**Use when:** General average, symmetric data
**Sensitive to outliers**

```python
mean([1, 2, 3, 4, 5])  # 3.0
```

### median
**Formula:** Middle value (sorted), average of two middle if n is even
**Use when:** Skewed data, outliers present
**Robust to outliers**

```python
median([1, 2, 100])  # 2 (not affected by outlier)
```

### mode
**Formula:** Most frequent value(s)
**Use when:** Categorical data, finding common values
**Returns list (multimodal possible)**

```python
mode([1, 2, 2, 3])  # [2]
```

## Dispersion

### variance / std
**Formula:** `σ² = Σ(xᵢ - μ)² / n` and `σ = √σ²`
**Use when:** Entire population available

```python
variance([1, 2, 3, 4, 5])  # 2.0
std([1, 2, 3, 4, 5])       # 1.414...
```

### sample_variance / sample_std
**Formula:** `s² = Σ(xᵢ - x̄)² / (n-1)` (Bessel's correction)
**Use when:** Data is sample from larger population
**Why n-1:** Corrects bias - sample tends to underestimate population variance

```python
sample_variance([1, 2, 3, 4, 5])  # 2.5
sample_std([1, 2, 3, 4, 5])       # 1.581...
```

### arr_range
**Formula:** `max - min`
**Use when:** Quick spread measure
**Sensitive to outliers**

### iqr
**Formula:** `Q3 - Q1` (75th - 25th percentile)
**Use when:** Robust spread measure needed

**Outlier detection:** Value is outlier if outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`

```python
iqr([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 4.0
```

## Shape

### skewness
**Formula:** `γ₁ = E[(X-μ)³] / σ³`

| Value | Interpretation |
|-------|----------------|
| > 0 | Right-skewed (tail right, mean > median) |
| < 0 | Left-skewed (tail left, mean < median) |
| ≈ 0 | Symmetric |

| Range | Degree |
|-------|--------|
| |γ| < 0.5 | Approximately symmetric |
| 0.5 ≤ |γ| < 1 | Moderately skewed |
| |γ| ≥ 1 | Highly skewed |

### kurtosis (4th moment)
**Formula:** `γ₂ = E[(X-μ)⁴] / σ⁴`
**Use:** `standardized_statistical_moment(arr, 4)`

| Value | Interpretation |
|-------|----------------|
| = 3 | Normal (mesokurtic) |
| > 3 | Heavy tails (leptokurtic) |
| < 3 | Light tails (platykurtic) |

## Choosing Measures

| Situation | Central Tendency | Spread |
|-----------|------------------|--------|
| Symmetric, no outliers | Mean | Std |
| Skewed or outliers | Median | IQR |
| Categorical | Mode | - |

## References
- [Khan Academy: Descriptive Statistics](https://www.khanacademy.org/math/statistics-probability)
- [Wikipedia: Descriptive Statistics](https://en.wikipedia.org/wiki/Descriptive_statistics)
