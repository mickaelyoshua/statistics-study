import math
from collections.abc import Sequence


def density_normal_distribution(std: float, mean: float, x: int | float) -> float:
    e = math.exp((-1 / 2) * ((x - mean) / std) ** 2)
    return (1 / (std * math.sqrt(2 * math.pi))) * e


def z_score(std: float, mean: float, x: int | float) -> float:
    return (x - mean) / std


if __name__ == "__main__":
    print("Hi")
