"""Normal distribution functions: PDF, CDF, and z-score transformations.

Functions:
    density_normal_distribution: PDF - probability density at a point
    z_score: Convert value to standard deviations from mean
    get_x_from_z_score: Convert z-score back to original value
    standard_normal_cdf: CDF for standard normal (mu=0, sigma=1)
    normal_cdf: CDF for any normal distribution
"""
import math


# Probability Density Function - The curve shape
def density_normal_distribution(sigma: float, mu: float, x: int | float) -> float:
    """PDF of normal distribution. sigma=std deviation, mu=mean."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    e = math.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
    return (1 / (sigma * math.sqrt(2 * math.pi))) * e


def z_score(sigma: float, mu: float, x: int | float) -> float:
    """Standardize a value: how many std deviations from the mean."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (x - mu) / sigma


def get_x_from_z_score(sigma: float, mu: float, z: float) -> float:
    """Convert z-score back to original value."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (z * sigma) + mu


# Cumulative Distribution Function - the accumulated area under the curve
# At any point equals the area under the PDF up to that point, just as in the table
def standard_normal_cdf(z: float) -> float:
    """Return P(Z <= z) for standard normal distribution (z-table value)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def normal_cdf(sigma: float, mu: float, x: int | float) -> float:
    """Return P(X <= x) for normal distribution with given sigma and mu."""
    z = z_score(sigma, mu, x)
    return standard_normal_cdf(z)


if __name__ == "__main__":
    mu = 27.0
    sigma = 2.5

    # First shoe length range: 27.1 - 27.6
    # Second shoe length range: 26.7 - 27.1
    upper_bound = normal_cdf(sigma, mu, 27.6)
    middle_bound = normal_cdf(sigma, mu, 27.1)
    lower_bound = normal_cdf(sigma, mu, 26.7)
    print(f"Percentage for x=27.6: {upper_bound}")
    print(f"Percentage for x=27.1: {middle_bound}")
    print(f"Percentage for x=26.7: {lower_bound}")
    print(
        f"Fist shoe length range percentage (27.1 to 27.6): {(upper_bound - middle_bound) * 100:.2f}%"
    )
    print(
        f"Second shoe length range percentage (26.7 to 27.1): {(middle_bound - lower_bound) * 100:.2f}%"
    )

    mu = 100
    sigma = 15
    print()
    print(
        f"Percentage of IQ range 85 to 115: {(normal_cdf(sigma, mu, 115) - normal_cdf(sigma, mu, 85)) * 100:.2f}"
    )
    print(
        f"Percentage for IQ of 121. This person is in the top: {(1 - normal_cdf(sigma, mu, 121))*100:.2f}"
    )
