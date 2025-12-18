from collections import Counter
from collections.abc import Sequence

#### CENTRAL TENDENCIES ####


def mean(arr: Sequence[int | float]) -> float:
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate mean from empty list.")
    return sum(arr) / n


def median(arr: Sequence[int | float]) -> float:
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
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate variance from empty list.")

    m = mean(arr)
    mean_centering = sum([(v - m) ** 2 for v in arr])
    return mean_centering / n


def std(arr: Sequence[int | float]) -> float:
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate standard deviation from empty list.")

    return variance(arr) ** (1 / 2)


def sample_variance(arr: Sequence[int | float]) -> float:
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate variance from empty list.")

    m = mean(arr)
    mean_centering = sum([(v - m) ** 2 for v in arr])
    return mean_centering / (n - 1)  # difference from variance is this n - 1


def sample_std(arr: Sequence[int | float]) -> float:
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate standard deviation from empty list.")

    return sample_variance(arr) ** (1 / 2)


def skewness(arr: Sequence[int | float]) -> float:
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate Skewness from empty list.")

    m = mean(arr)
    # The powr of 3 to make outcast more significant and can have negative values
    mean_centering = sum([(v - m) ** 3 for v in arr])

    return (mean_centering / n) / (std(arr) ** 3)
    # if result is > 0 then is positive skew (right-skewed)
    # if result is < 0 then is negative skew (left-skewed)


def arr_range(arr: Sequence[int | float]) -> float:
    if len(arr) == 0:
        raise ValueError("Can not calculate Range from empty list.")

    return max(arr) - min(arr)


def iqr(arr: Sequence[int | float]) -> float:
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate IQR from empty list.")

    sorted_arr = sorted(arr)

    if n % 2 == 0:
        return median(sorted_arr[n // 2 :]) - median(sorted_arr[: n // 2])
    else:
        i = n // 2  # floor division
        return median(sorted_arr[(i + 1) :]) - median(sorted_arr[:i])


def standardized_statistical_moment(arr: Sequence[int | float], moment: int) -> float:
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

    m = mean(arr)
    mean_centering = sum([(v - m) ** moment for v in arr])

    return (mean_centering / n) / (std(arr) ** moment)


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
