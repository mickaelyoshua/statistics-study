"""Normal distribution functions: PDF, CDF, and z-score transformations.

The normal (Gaussian) distribution is defined by two parameters:
- μ (mu): Mean - center of the distribution
- σ (sigma): Standard deviation - spread of the distribution

Functions:
    normal_pdf: Probability density at a point
    z_score: Convert value to standard deviations from mean
    get_x_from_z_score: Convert z-score back to original value
    standard_normal_cdf: CDF for standard normal (μ=0, σ=1)
    normal_cdf: CDF for any normal distribution

The 68-95-99.7 rule (empirical rule):
- ~68% of data falls within 1 standard deviation of the mean
- ~95% of data falls within 2 standard deviations
- ~99.7% of data falls within 3 standard deviations
"""

import math


# =============================================================================
# Constants
# =============================================================================

_SQRT_2_PI = math.sqrt(2 * math.pi)
_SQRT_2 = math.sqrt(2)


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_sigma(sigma: float) -> None:
    """Validate that sigma (standard deviation) is positive.

    Args:
        sigma: Standard deviation to validate

    Raises:
        ValueError: If sigma <= 0
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")


# =============================================================================
# Probability Density Function (PDF)
# =============================================================================


def normal_pdf(sigma: float, mu: float, x: float) -> float:
    """Calculate probability density at x for normal distribution.

    The PDF gives the relative likelihood of a value. For continuous
    distributions, the probability at any exact point is technically 0;
    the PDF indicates density, not probability.

    Formula: f(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))

    Args:
        sigma: Standard deviation (σ) - must be positive
        mu: Mean (μ) - center of distribution
        x: Value at which to calculate density

    Returns:
        Probability density at x

    Raises:
        ValueError: If sigma <= 0

    Example:
        >>> normal_pdf(1, 0, 0)  # Standard normal at mean
        0.3989422804014327
    """
    _validate_sigma(sigma)
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return (1 / (sigma * _SQRT_2_PI)) * math.exp(exponent)


# Alias for backward compatibility
density_normal_distribution = normal_pdf


# =============================================================================
# Z-Score Transformations
# =============================================================================


def z_score(sigma: float, mu: float, x: float) -> float:
    """Convert a value to its z-score (standard score).

    The z-score indicates how many standard deviations a value is
    from the mean. Used to standardize values from different distributions.

    Formula: z = (x - μ) / σ

    Args:
        sigma: Standard deviation (σ) of the distribution
        mu: Mean (μ) of the distribution
        x: Value to standardize

    Returns:
        Z-score (number of standard deviations from mean)

    Raises:
        ValueError: If sigma <= 0

    Example:
        >>> z_score(15, 100, 130)  # IQ of 130 with mean 100, std 15
        2.0
    """
    _validate_sigma(sigma)
    return (x - mu) / sigma


def get_x_from_z_score(sigma: float, mu: float, z: float) -> float:
    """Convert a z-score back to the original value.

    Inverse of z_score function.

    Formula: x = μ + z * σ

    Args:
        sigma: Standard deviation (σ) of the distribution
        mu: Mean (μ) of the distribution
        z: Z-score to convert

    Returns:
        Original value corresponding to the z-score

    Raises:
        ValueError: If sigma <= 0

    Example:
        >>> get_x_from_z_score(15, 100, 2.0)  # z=2 with mean 100, std 15
        130.0
    """
    _validate_sigma(sigma)
    return mu + (z * sigma)


# =============================================================================
# Cumulative Distribution Function (CDF)
# =============================================================================


def standard_normal_cdf(z: float) -> float:
    """Calculate P(Z <= z) for standard normal distribution.

    Returns the cumulative probability up to z for the standard normal
    distribution (μ=0, σ=1). This is the value you would look up in a z-table.

    Uses the error function (erf) for precise calculation.
    Formula: Φ(z) = 0.5 * (1 + erf(z/√2))

    Args:
        z: Z-score value

    Returns:
        Probability that Z <= z (value between 0 and 1)

    Example:
        >>> standard_normal_cdf(0)    # P(Z <= 0)
        0.5
        >>> standard_normal_cdf(1.96)  # ~97.5% (two-tailed 95% CI)
        0.9750021...
    """
    return 0.5 * (1 + math.erf(z / _SQRT_2))


def normal_cdf(sigma: float, mu: float, x: float) -> float:
    """Calculate P(X <= x) for normal distribution.

    Returns the cumulative probability up to x for a normal distribution
    with specified mean and standard deviation.

    Args:
        sigma: Standard deviation (σ) - must be positive
        mu: Mean (μ) of the distribution
        x: Value to calculate cumulative probability for

    Returns:
        Probability that X <= x (value between 0 and 1)

    Raises:
        ValueError: If sigma <= 0

    Example:
        >>> normal_cdf(15, 100, 115)  # P(IQ <= 115)
        0.8413...
    """
    z = z_score(sigma, mu, x)
    return standard_normal_cdf(z)


# =============================================================================
# Main - Examples
# =============================================================================


if __name__ == "__main__":
    print("=== Normal Distribution Examples ===\n")

    # Shoe length example
    mu = 27.0
    sigma = 2.5
    print(f"Shoe length distribution: μ={mu}, σ={sigma}")

    upper = normal_cdf(sigma, mu, 27.6)
    middle = normal_cdf(sigma, mu, 27.1)
    lower = normal_cdf(sigma, mu, 26.7)

    print(f"P(X ≤ 27.6): {upper:.4f}")
    print(f"P(X ≤ 27.1): {middle:.4f}")
    print(f"P(X ≤ 26.7): {lower:.4f}")
    print(f"P(27.1 < X ≤ 27.6): {(upper - middle) * 100:.2f}%")
    print(f"P(26.7 < X ≤ 27.1): {(middle - lower) * 100:.2f}%")

    # IQ example
    print("\n--- IQ Distribution (μ=100, σ=15) ---")
    mu = 100
    sigma = 15

    # Empirical rule demonstration
    within_1_std = normal_cdf(sigma, mu, 115) - normal_cdf(sigma, mu, 85)
    print(f"P(85 < IQ ≤ 115) [within 1σ]: {within_1_std * 100:.2f}%")

    # Percentile calculation
    iq_121 = 1 - normal_cdf(sigma, mu, 121)
    print(f"P(IQ > 121) [top percentile]: {iq_121 * 100:.2f}%")
