"""Hypothesis testing: errors, p-values, confidence intervals, and statistical tests.

Key Concepts:
    Type I Error (α): Rejecting true H₀ (false positive)
    Type II Error (β): Failing to reject false H₀ (false negative)
    Power = 1 - β: Probability of correctly rejecting false H₀
    p-value: P(observing result | H₀ true)

Decision Rule:
    If p-value < α → reject H₀
    If p-value ≥ α → fail to reject H₀

Common α values: 0.01, 0.05, 0.10
Common confidence levels: 90%, 95%, 99%

Functions:
    standard_error: SE = σ/√n
    margin_of_error: ME = z * SE
    confidence_interval: x̄ ± ME
    z_test_statistic: z = (x̄ - μ₀) / SE
    p_value_from_z: Convert z-score to p-value
    is_significant: Check if p < α
    probability_within_n_std: P(μ-nσ < X < μ+nσ)
"""

import math

from stats.distributions import standard_normal_cdf


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_positive_int(n: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_probability(p: float, name: str) -> None:
    """Validate that a probability is in (0, 1)."""
    if not 0 < p < 1:
        raise ValueError(f"{name} must be between 0 and 1 (exclusive)")


def _validate_confidence(confidence: float) -> None:
    """Validate confidence level is in (0, 1)."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1 (e.g., 0.95)")


# =============================================================================
# Critical Values
# =============================================================================

# Common z-critical values for quick lookup
Z_CRITICAL = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


def z_critical(confidence: float) -> float:
    """Get z-critical value for a confidence level.

    For two-tailed tests, this is the z-value where:
    P(-z < Z < z) = confidence

    Formula: z = Φ⁻¹((1 + confidence) / 2)

    Args:
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Z-critical value

    Raises:
        ValueError: If confidence not in (0, 1)

    Example:
        >>> z_critical(0.95)
        1.96
    """
    _validate_confidence(confidence)

    # Use lookup for common values
    if confidence in Z_CRITICAL:
        return Z_CRITICAL[confidence]

    # Approximate using inverse error function
    # z = √2 * erf⁻¹(confidence)
    alpha = 1 - confidence
    # For two-tailed: we need the z where P(Z > z) = α/2
    # Using approximation based on rational function
    p = 1 - alpha / 2

    # Abramowitz and Stegun approximation for inverse normal CDF
    if p <= 0 or p >= 1:
        raise ValueError("Invalid probability for z_critical calculation")

    # Rational approximation constants
    a = [0, -3.969683028665376e1, 2.209460984245205e2,
         -2.759285104469687e2, 1.383577518672690e2,
         -3.066479806614716e1, 2.506628277459239e0]
    b = [0, -5.447609879822406e1, 1.615858368580409e2,
         -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
    c = [0, -7.784894002430293e-3, -3.223964580411365e-1,
         -2.400758277161838, -2.549732539343734,
         4.374664141464968, 2.938163982698783]
    d = [0, 7.784695709041462e-3, 3.224671290700398e-1,
         2.445134137142996, 3.754408661907416]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) / \
            ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6])*q / \
            (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) / \
            ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)

    return round(z, 3)


# =============================================================================
# Standard Error and Margin of Error
# =============================================================================


def standard_error(sigma: float, n: int) -> float:
    """Calculate standard error of the sample mean.

    SE measures how much sample means vary from the true population mean.
    Decreases as sample size increases.

    Formula: SE = σ / √n

    Args:
        sigma: Population standard deviation
        n: Sample size

    Returns:
        Standard error

    Raises:
        ValueError: If sigma ≤ 0 or n ≤ 0

    Example:
        >>> standard_error(sigma=10, n=100)
        1.0
    """
    _validate_positive(sigma, "sigma")
    _validate_positive_int(n, "n")
    return sigma / math.sqrt(n)


def standard_error_proportion(p: float, n: int) -> float:
    """Calculate standard error for a sample proportion.

    Formula: SE = √(p(1-p)/n)

    Args:
        p: Sample proportion (between 0 and 1)
        n: Sample size

    Returns:
        Standard error of proportion

    Raises:
        ValueError: If p not in (0,1) or n ≤ 0

    Example:
        >>> standard_error_proportion(p=0.5, n=100)
        0.05
    """
    _validate_probability(p, "p")
    _validate_positive_int(n, "n")
    return math.sqrt(p * (1 - p) / n)


def margin_of_error(sigma: float, n: int, confidence: float = 0.95) -> float:
    """Calculate margin of error for confidence interval.

    ME is half the width of the confidence interval.

    Formula: ME = z * (σ / √n)

    Args:
        sigma: Population standard deviation
        n: Sample size
        confidence: Confidence level (default 0.95)

    Returns:
        Margin of error

    Raises:
        ValueError: If sigma ≤ 0, n ≤ 0, or confidence invalid

    Example:
        >>> margin_of_error(sigma=10, n=100, confidence=0.95)
        1.96
    """
    _validate_positive(sigma, "sigma")
    _validate_positive_int(n, "n")
    _validate_confidence(confidence)

    z = z_critical(confidence)
    se = standard_error(sigma, n)
    return z * se


# =============================================================================
# Confidence Intervals
# =============================================================================


def confidence_interval(
    x_bar: float, sigma: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate confidence interval for population mean.

    A 95% CI means: if we repeated sampling many times, 95% of
    intervals would contain the true population mean.

    Formula: x̄ ± z * (σ / √n)

    Args:
        x_bar: Sample mean
        sigma: Population standard deviation
        n: Sample size
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Raises:
        ValueError: If sigma ≤ 0, n ≤ 0, or confidence invalid

    Example:
        >>> confidence_interval(x_bar=100, sigma=15, n=100, confidence=0.95)
        (97.06, 102.94)
    """
    me = margin_of_error(sigma, n, confidence)
    return (round(x_bar - me, 2), round(x_bar + me, 2))


# =============================================================================
# Hypothesis Testing
# =============================================================================


def z_test_statistic(x_bar: float, mu_0: float, sigma: float, n: int) -> float:
    """Calculate z-test statistic for testing population mean.

    Use when: σ is known AND n ≥ 30

    H₀: μ = μ₀ (null hypothesis)
    H₁: μ ≠ μ₀ (two-tailed) or μ > μ₀ or μ < μ₀ (one-tailed)

    Formula: z = (x̄ - μ₀) / (σ / √n)

    Args:
        x_bar: Sample mean
        mu_0: Hypothesized population mean (null hypothesis)
        sigma: Population standard deviation
        n: Sample size

    Returns:
        Z-test statistic

    Raises:
        ValueError: If sigma ≤ 0 or n ≤ 0

    Example:
        >>> z_test_statistic(x_bar=105, mu_0=100, sigma=15, n=100)
        3.333...
    """
    _validate_positive(sigma, "sigma")
    _validate_positive_int(n, "n")

    se = standard_error(sigma, n)
    return (x_bar - mu_0) / se


def p_value_from_z(z: float, alternative: str = "two-sided") -> float:
    """Calculate p-value from z-statistic.

    P-value: probability of observing a result at least as extreme
    as the one obtained, assuming H₀ is true.

    Args:
        z: Z-test statistic
        alternative: Type of test
            - "two-sided": H₁: μ ≠ μ₀
            - "greater": H₁: μ > μ₀
            - "less": H₁: μ < μ₀

    Returns:
        P-value

    Raises:
        ValueError: If alternative is invalid

    Example:
        >>> p_value_from_z(1.96, "two-sided")
        0.05
        >>> p_value_from_z(1.96, "greater")
        0.025
    """
    valid_alternatives = ("two-sided", "greater", "less")
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}")

    if alternative == "two-sided":
        # P(|Z| > |z|) = 2 * P(Z > |z|)
        return 2 * (1 - standard_normal_cdf(abs(z)))
    elif alternative == "greater":
        # P(Z > z)
        return 1 - standard_normal_cdf(z)
    else:  # less
        # P(Z < z)
        return standard_normal_cdf(z)


def is_significant(p_value: float, alpha: float = 0.05) -> bool:
    """Check if result is statistically significant.

    Decision rule:
        - p < α → Reject H₀ (significant)
        - p ≥ α → Fail to reject H₀ (not significant)

    Args:
        p_value: P-value from test
        alpha: Significance level (default 0.05)

    Returns:
        True if statistically significant

    Example:
        >>> is_significant(0.03, alpha=0.05)
        True
        >>> is_significant(0.10, alpha=0.05)
        False
    """
    return p_value < alpha


# =============================================================================
# Probability Within Range
# =============================================================================


def probability_within_n_std(
    mu: float, sigma: float, n: float
) -> tuple[float, float, float]:
    """Calculate probability of value within n standard deviations.

    Uses the property that for normal distribution:
    P(μ - nσ < X < μ + nσ) = 2Φ(n) - 1

    Common results (empirical rule):
        n=1: 68.27%
        n=2: 95.45%
        n=3: 99.73%

    Args:
        mu: Mean of distribution
        sigma: Standard deviation
        n: Number of standard deviations

    Returns:
        Tuple of (lower_bound, upper_bound, probability)

    Example:
        >>> probability_within_n_std(mu=100, sigma=15, n=1)
        (85, 115, 0.6827)
    """
    _validate_positive(sigma, "sigma")
    _validate_positive(n, "n")

    lower = mu - n * sigma
    upper = mu + n * sigma
    prob = 2 * standard_normal_cdf(n) - 1
    return (lower, upper, round(prob, 4))


# =============================================================================
# Main - Examples
# =============================================================================


if __name__ == "__main__":
    print("=== Hypothesis Testing Examples ===\n")

    # Confidence interval example
    print("--- Confidence Interval ---")
    x_bar, sigma, n = 98.6, 0.7, 50
    ci = confidence_interval(x_bar, sigma, n, 0.95)
    print(f"Sample: x̄={x_bar}, σ={sigma}, n={n}")
    print(f"95% CI: {ci}")
    print()

    # Z-test example
    print("--- Z-Test ---")
    x_bar, mu_0, sigma, n = 105, 100, 15, 100
    z = z_test_statistic(x_bar, mu_0, sigma, n)
    p = p_value_from_z(z, "two-sided")
    print(f"H₀: μ = {mu_0}")
    print(f"Sample: x̄={x_bar}, σ={sigma}, n={n}")
    print(f"z-statistic: {z:.4f}")
    print(f"p-value (two-sided): {p:.4f}")
    print(f"Significant at α=0.05? {is_significant(p, 0.05)}")
    print()

    # Probability within n std
    print("--- Empirical Rule ---")
    for n_std in [1, 2, 3]:
        lower, upper, prob = probability_within_n_std(100, 15, n_std)
        print(f"{n_std}σ: {prob*100:.2f}% within [{lower}, {upper}]")
