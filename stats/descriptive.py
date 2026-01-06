"""Descriptive statistics: measures of central tendency and dispersion.

Central Tendencies:
    mean: Arithmetic average - sum of values divided by count
    median: Middle value when sorted - robust to outliers
    mode: Most frequent value(s) - can have multiple modes

Dispersion (Spread):
    variance/std: Population measures (divide by n)
    sample_variance/sample_std: Sample measures (divide by n-1, Bessel's correction)
    skewness: Asymmetry of distribution (>0 right-skewed, <0 left-skewed)
    arr_range: Difference between max and min
    iqr: Interquartile range (Q3 - Q1) - robust to outliers
    standardized_statistical_moment: Generalized moment calculation

Population vs Sample:
    Use population measures when you have data for the entire population.
    Use sample measures when your data is a sample from a larger population.
    Bessel's correction (n-1) provides an unbiased estimate of population variance.
"""

import math
from collections import Counter
from collections.abc import Sequence


# =============================================================================
# Central Tendencies
# =============================================================================


def mean(arr: Sequence[int | float]) -> float:
    """Calculate arithmetic mean (average) of values.

    Formula: x̄ = (Σxᵢ) / n

    Args:
        arr: Sequence of numeric values

    Returns:
        Arithmetic mean of the values

    Raises:
        ValueError: If arr is empty

    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot calculate mean from empty list")
    return sum(arr) / n


def median(arr: Sequence[int | float]) -> float:
    """Calculate median (middle value when sorted).

    For even n, returns the average of the two middle values.
    More robust to outliers than mean.

    Args:
        arr: Sequence of numeric values

    Returns:
        Median value

    Raises:
        ValueError: If arr is empty

    Example:
        >>> median([1, 2, 3, 4, 5])
        3
        >>> median([1, 2, 3, 4])
        2.5
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot calculate median from empty list")

    sorted_arr = sorted(arr)

    if n % 2 == 0:
        i1 = n // 2
        i2 = i1 - 1
        return (sorted_arr[i1] + sorted_arr[i2]) / 2
    else:
        return sorted_arr[n // 2]


def mode(arr: Sequence[int | float]) -> list[int | float]:
    """Find mode(s) - most frequent value(s).

    Returns a list since multiple modes are possible (multimodal distribution).

    Args:
        arr: Sequence of numeric values

    Returns:
        List of mode value(s)

    Raises:
        ValueError: If arr is empty
        ValueError: If all values have the same frequency (no mode)

    Example:
        >>> mode([1, 2, 2, 3])
        [2]
        >>> mode([1, 1, 2, 2])
        [1, 2]
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot calculate mode from empty list")

    counter = Counter(arr)
    max_freq = max(counter.values())
    if max_freq == 1:
        raise ValueError("No mode: all values have the same frequency")

    modes = [k for k, v in counter.items() if v == max_freq]
    return modes


# =============================================================================
# Dispersion (Population Measures)
# =============================================================================


def variance(arr: Sequence[int | float]) -> float:
    """Calculate population variance.

    Sum of squared deviations from the mean, divided by n.
    Use when data represents the entire population.

    Formula: σ² = Σ(xᵢ - μ)² / n

    Args:
        arr: Sequence of numeric values

    Returns:
        Population variance

    Raises:
        ValueError: If arr is empty

    Example:
        >>> variance([2, 4, 4, 4, 5, 5, 7, 9])
        4.0
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot calculate variance from empty list")

    m = mean(arr)
    return sum((v - m) ** 2 for v in arr) / n


def std(arr: Sequence[int | float]) -> float:
    """Calculate population standard deviation.

    Square root of population variance. Same units as original data.

    Formula: σ = √(Σ(xᵢ - μ)² / n)

    Args:
        arr: Sequence of numeric values

    Returns:
        Population standard deviation

    Raises:
        ValueError: If arr is empty

    Example:
        >>> std([2, 4, 4, 4, 5, 5, 7, 9])
        2.0
    """
    if len(arr) == 0:
        raise ValueError("Cannot calculate standard deviation from empty list")

    return math.sqrt(variance(arr))


# =============================================================================
# Dispersion (Sample Measures - Bessel's Correction)
# =============================================================================


def sample_variance(arr: Sequence[int | float]) -> float:
    """Calculate sample variance using Bessel's correction.

    Sum of squared deviations divided by (n-1) instead of n.
    Provides unbiased estimate of population variance from a sample.

    Formula: s² = Σ(xᵢ - x̄)² / (n-1)

    Args:
        arr: Sequence of numeric values (at least 2 values)

    Returns:
        Sample variance

    Raises:
        ValueError: If arr is empty
        ValueError: If arr has fewer than 2 values

    Example:
        >>> sample_variance([2, 4, 4, 4, 5, 5, 7, 9])
        4.571428571428571
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot calculate variance from empty list")
    if n < 2:
        raise ValueError("Sample variance requires at least 2 values")

    m = mean(arr)
    return sum((v - m) ** 2 for v in arr) / (n - 1)


def sample_std(arr: Sequence[int | float]) -> float:
    """Calculate sample standard deviation.

    Square root of sample variance.

    Formula: s = √(Σ(xᵢ - x̄)² / (n-1))

    Args:
        arr: Sequence of numeric values (at least 2 values)

    Returns:
        Sample standard deviation

    Raises:
        ValueError: If arr is empty
        ValueError: If arr has fewer than 2 values

    Example:
        >>> sample_std([2, 4, 4, 4, 5, 5, 7, 9])
        2.138089935299395
    """
    if len(arr) == 0:
        raise ValueError("Cannot calculate standard deviation from empty list")
    if len(arr) < 2:
        raise ValueError("Sample standard deviation requires at least 2 values")

    return math.sqrt(sample_variance(arr))


# =============================================================================
# Shape and Range Measures
# =============================================================================


def skewness(arr: Sequence[int | float]) -> float:
    """Calculate skewness (3rd standardized moment).

    Measures the asymmetry of a distribution.

    Formula: γ₁ = E[(X-μ)³] / σ³

    Args:
        arr: Sequence of numeric values

    Returns:
        Skewness coefficient:
        - > 0: Right-skewed (tail extends right, mean > median)
        - < 0: Left-skewed (tail extends left, mean < median)
        - ≈ 0: Symmetric distribution

    Raises:
        ValueError: If arr is empty
        ValueError: If all values are identical (std = 0)

    Example:
        >>> skewness([1, 2, 2, 3, 3, 3, 4, 4, 5])
        0.0
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot calculate skewness from empty list")

    s = std(arr)
    if s == 0:
        raise ValueError("Cannot calculate skewness when all values are identical")

    m = mean(arr)
    return sum((v - m) ** 3 for v in arr) / n / (s ** 3)


def arr_range(arr: Sequence[int | float]) -> float:
    """Calculate range: max - min.

    Simple measure of spread. Sensitive to outliers.

    Args:
        arr: Sequence of numeric values

    Returns:
        Difference between maximum and minimum values

    Raises:
        ValueError: If arr is empty

    Example:
        >>> arr_range([1, 5, 3, 9, 2])
        8
    """
    if len(arr) == 0:
        raise ValueError("Cannot calculate range from empty list")

    return max(arr) - min(arr)


def iqr(arr: Sequence[int | float]) -> float:
    """Calculate interquartile range (Q3 - Q1).

    Measures the spread of the middle 50% of data.
    More robust to outliers than range.

    Args:
        arr: Sequence of numeric values (at least 2 values)

    Returns:
        Difference between 75th and 25th percentiles

    Raises:
        ValueError: If arr is empty
        ValueError: If arr has fewer than 2 values

    Example:
        >>> iqr([1, 2, 3, 4, 5, 6, 7, 8, 9])
        4.0
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot calculate IQR from empty list")
    if n < 2:
        raise ValueError("IQR requires at least 2 values")

    sorted_arr = sorted(arr)

    if n % 2 == 0:
        return median(sorted_arr[n // 2:]) - median(sorted_arr[:n // 2])
    else:
        i = n // 2
        return median(sorted_arr[i + 1:]) - median(sorted_arr[:i])


# =============================================================================
# Generalized Moments
# =============================================================================


def standardized_statistical_moment(arr: Sequence[int | float], moment: int) -> float:
    """Calculate standardized moment of given order.

    Standardized moments are normalized by standard deviation to be
    scale-independent.

    Formula: μₖ = E[(X-μ)ᵏ] / σᵏ

    Special cases:
        - Moment 0: Always 1 (by definition)
        - Moment 1: Always 0 (mean-centered)
        - Moment 2: Always 1 (normalized by variance)
        - Moment 3: Skewness (asymmetry)
        - Moment 4: Kurtosis (tail heaviness)

    Args:
        arr: Sequence of numeric values
        moment: Order of the moment (must be >= 0)

    Returns:
        Standardized moment of the specified order

    Raises:
        ValueError: If arr is empty
        ValueError: If moment < 0
        ValueError: If all values are identical (for moment > 2)

    Example:
        >>> standardized_statistical_moment([1, 2, 3, 4, 5], 3)
        0.0
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot calculate statistical moment from empty list")
    if moment < 0:
        raise ValueError("Cannot calculate statistical moment for moment less than 0")

    # Special cases
    if moment == 0:
        return 1.0
    if moment == 1:
        return 0.0
    if moment == 2:
        return 1.0

    s = std(arr)
    if s == 0:
        raise ValueError(
            "Cannot calculate statistical moment when all values are identical"
        )

    m = mean(arr)
    return sum((v - m) ** moment for v in arr) / n / (s ** moment)


if __name__ == "__main__":
    arr = [13, 15, 17, 14, 15, 9, 8]
    arr2 = [1, 2, 2, 1, 3, 4, 3, 2, 5, 1, 2, 7, 3, 4, 5, 6, 2, 1]
    arr3 = [2, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7]
    arr4 = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6]
    print(f"Mean   {mean(arr)}")
    print(f"Median {median(arr)}")
    print(f"Mode   {mode(arr)}")
    print(f"Sample Variance {sample_variance(arr)}")
    print(f"Skewness {skewness(arr2)}")
    print(f"IQR (n is odd) {iqr(arr3)}")
    print(f"IQR (n is even) {iqr(arr4)}")
