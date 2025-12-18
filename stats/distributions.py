import math
from collections.abc import Sequence

from stats.descriptive import mean, std, variance


def density_normal_distribution(sigma: float, mu: float, x: int | float) -> float:
    """PDF of normal distribution. sigma=std deviation, mu=mean."""
    e = math.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
    return (1 / (sigma * math.sqrt(2 * math.pi))) * e


def z_score(sigma: float, mu: float, x: int | float) -> float:
    """Standardize a value: how many std deviations from the mean."""
    return (x - mu) / sigma


def get_x_from_z_score(sigma: float, mu: float, z: float) -> float:
    """Convert z-score back to original value."""
    return (z * sigma) + mu


if __name__ == "__main__":
    print("Hi")
