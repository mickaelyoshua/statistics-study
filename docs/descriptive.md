# Descriptive Statistics

Summarize and describe data characteristics.

## Central Tendencies

### mean
**Formula:** `x̄ = Σxᵢ / n`
**Use when:** General average needed, data is symmetric
**Sensitive to outliers**

```python
mean([1, 2, 3, 4, 5])  # 3.0
```

### median
**Formula:** Middle value when sorted (average of two middle if n is even)
**Use when:** Data has outliers, skewed distribution
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
**Use when:** You have the entire population

```python
variance([1, 2, 3, 4, 5])  # 2.0
std([1, 2, 3, 4, 5])       # 1.414...
```

### sample_variance / sample_std
**Formula:** `s² = Σ(xᵢ - x̄)² / (n-1)` (Bessel's correction)
**Use when:** Data is a sample from larger population

```python
sample_variance([1, 2, 3, 4, 5])  # 2.5
sample_std([1, 2, 3, 4, 5])       # 1.581...
```

### arr_range
**Formula:** `max - min`
**Use when:** Quick spread measure needed
**Sensitive to outliers**

### iqr
**Formula:** `Q3 - Q1` (75th - 25th percentile)
**Use when:** Robust spread measure needed
**Used for outlier detection:** values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`

## Shape

### skewness
**Formula:** `γ₁ = E[(X-μ)³] / σ³`
**Interpretation:**
- `> 0`: Right-skewed (tail right, mean > median)
- `< 0`: Left-skewed (tail left, mean < median)
- `≈ 0`: Symmetric

```python
skewness([1, 1, 1, 2, 10])  # Positive (right-skewed)
```

## References
- [Khan Academy: Descriptive Statistics](https://www.khanacademy.org/math/statistics-probability)
- [Wikipedia: Descriptive Statistics](https://en.wikipedia.org/wiki/Descriptive_statistics)
