"""Descriptive statistics: measures of central tendency and dispersion.

Central Tendencies:
    mean: Arithmetic average
    median: Middle value when sorted
    mode: Most frequent value(s)

Dispersion:
    variance/std: Population measures (divide by n)
    sample_variance/sample_std: Sample measures (divide by n-1, Bessel's correction)
    skewness: Asymmetry of distribution (>0 right-skewed, <0 left-skewed)
    arr_range: Difference between max and min
    iqr: Interquartile range (Q3 - Q1)
    standardized_statistical_moment: Generalized moment calculation
"""
from collections import Counter
from collections.abc import Sequence

#### CENTRAL TENDENCIES ####


def mean(arr: Sequence[int | float]) -> float:
    """Calculate arithmetic mean (average) of values."""
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate mean from empty list.")
    return sum(arr) / n


def median(arr: Sequence[int | float]) -> float:
    """Calculate median (middle value when sorted). Averages two middle values if n is even."""
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate median from empty list.")

    sorted_arr = sorted(arr)

    if n % 2 == 0:
        i1 = n // 2  # floor division
        i2 = i1 - 1
        return mean([sorted_arr[i1], sorted_arr[i2]])
    else:
        return sorted_arr[n // 2]


def mode(arr: Sequence[int | float]) -> list[int | float]:
    """Find mode(s) - most frequent value(s). Returns list since multiple modes possible."""
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate mode from empty list.")

    counter = Counter(arr)
    max_freq = max(counter.values())
    if max_freq == 1:
        raise ValueError("No mode: all values have the same frequency")

    modes = [k for k, v in counter.items() if v == max_freq]
    return modes


#### DISPERSION ####


def variance(arr: Sequence[int | float]) -> float:
    """Calculate population variance. Sum of squared deviations divided by n."""
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate variance from empty list.")

    m = mean(arr)
    mean_centering = sum([(v - m) ** 2 for v in arr])
    return mean_centering / n


def std(arr: Sequence[int | float]) -> float:
    """Calculate population standard deviation. Square root of variance."""
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate standard deviation from empty list.")

    return variance(arr) ** (1 / 2)


def sample_variance(arr: Sequence[int | float]) -> float:
    """Calculate sample variance using Bessel's correction (n-1 denominator)."""
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate variance from empty list.")
    if n < 2:
        raise ValueError("Sample variance requires at least 2 values.")

    m = mean(arr)
    mean_centering = sum([(v - m) ** 2 for v in arr])
    return mean_centering / (n - 1)


def sample_std(arr: Sequence[int | float]) -> float:
    """Calculate sample standard deviation. Square root of sample variance."""
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate standard deviation from empty list.")
    if n < 2:
        raise ValueError("Sample standard deviation requires at least 2 values.")

    return sample_variance(arr) ** (1 / 2)


def skewness(arr: Sequence[int | float]) -> float:
    """Calculate skewness (3rd standardized moment). Measures distribution asymmetry.

    Returns:
        >0: Right-skewed (tail extends right)
        <0: Left-skewed (tail extends left)
        ~0: Symmetric
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate Skewness from empty list.")

    s = std(arr)
    if s == 0:
        raise ValueError("Can not calculate Skewness when all values are identical.")

    m = mean(arr)
    mean_centering = sum([(v - m) ** 3 for v in arr])

    return (mean_centering / n) / (s ** 3)


def arr_range(arr: Sequence[int | float]) -> float:
    """Calculate range: max - min."""
    if len(arr) == 0:
        raise ValueError("Can not calculate Range from empty list.")

    return max(arr) - min(arr)


def iqr(arr: Sequence[int | float]) -> float:
    """Calculate interquartile range (Q3 - Q1). Measures spread of middle 50% of data."""
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate IQR from empty list.")
    if n < 2:
        raise ValueError("IQR requires at least 2 values.")

    sorted_arr = sorted(arr)

    if n % 2 == 0:
        return median(sorted_arr[n // 2 :]) - median(sorted_arr[: n // 2])
    else:
        i = n // 2
        return median(sorted_arr[(i + 1) :]) - median(sorted_arr[:i])


def standardized_statistical_moment(arr: Sequence[int | float], moment: int) -> float:
    """Calculate standardized moment of given order.

    Moments 0, 1, 2 always return 1, 0, 1 respectively.
    Moment 3 = skewness, Moment 4 = kurtosis.
    """
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate Statistical Moment from empty list.")
    if moment < 0:
        raise ValueError(
            "Can not calculate Statistical Moment from moment less then 0."
        )

    if moment == 0:
        return 1.0
    if moment == 1:
        return 0.0
    if moment == 2:
        return 1.0

    s = std(arr)
    if s == 0:
        raise ValueError("Can not calculate Statistical Moment when all values are identical.")

    m = mean(arr)
    mean_centering = sum([(v - m) ** moment for v in arr])

    return (mean_centering / n) / (s ** moment)


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
