from collections.abc import Sequence  # for any sequence type


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


def mode(arr: Sequence[int | float]) -> int | float:
    n = len(arr)
    if n == 0:
        raise ValueError("Can not calculate median from empty list.")


if __name__ == "__main__":
    arr = [13, 15, 17, 14, 15, 9, 8]
    print(f"Mean {mean(arr)}")
    print(f"Median {median(arr)}")
