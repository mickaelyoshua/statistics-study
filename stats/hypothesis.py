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

Z-Test Functions (σ known, n ≥ 30):
    z_critical: Get z-value for confidence level
    z_test_statistic: z = (x̄ - μ₀) / (σ/√n)
    p_value_from_z: Convert z-score to p-value
    confidence_interval: x̄ ± z * (σ/√n)

T-Test Functions (σ unknown or n < 30):
    t_cdf: Cumulative distribution function for t-distribution
    t_critical: Get t-value for confidence level and df
    t_test_statistic: t = (x̄ - μ₀) / (s/√n)
    p_value_from_t: Convert t-statistic to p-value
    confidence_interval_t: x̄ ± t * (s/√n)

Common Functions:
    standard_error: SE = σ/√n
    margin_of_error: ME = z * SE
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
# T-Distribution Functions
# =============================================================================


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Calculate regularized incomplete beta function I_x(a, b).

    Uses continued fraction expansion for numerical stability.
    This is used internally for t-distribution CDF calculation.

    Args:
        a: First shape parameter (positive)
        b: Second shape parameter (positive)
        x: Upper limit of integration (0 <= x <= 1)

    Returns:
        Value of regularized incomplete beta function
    """
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Use symmetry relation if x > (a+1)/(a+b+2)
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(b, a, 1.0 - x)

    # Continued fraction expansion (Lentz's algorithm)
    max_iter = 200
    epsilon = 1e-10

    # Calculate the prefactor
    ln_prefactor = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    prefactor = math.exp(ln_prefactor) / a

    # Continued fraction
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    result = d

    for m in range(1, max_iter + 1):
        # Even step
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        result *= d * c

        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        result *= delta

        if abs(delta - 1.0) < epsilon:
            break

    return prefactor * result


def t_cdf(t: float, df: int) -> float:
    """Calculate cumulative distribution function for t-distribution.

    Returns P(T ≤ t) for Student's t-distribution with df degrees of freedom.
    This replaces looking up values in a t-table.

    Formula: Uses regularized incomplete beta function

    Args:
        t: T-statistic value
        df: Degrees of freedom (must be positive integer)

    Returns:
        Probability P(T ≤ t), value between 0 and 1

    Raises:
        ValueError: If df ≤ 0

    Example:
        >>> t_cdf(0, 10)      # t=0 always gives 0.5
        0.5
        >>> t_cdf(2.228, 10)  # 97.5th percentile for df=10
        0.975
    """
    _validate_positive_int(df, "df")

    # Convert to regularized incomplete beta function
    x = df / (df + t * t)

    if t >= 0:
        return 1.0 - 0.5 * _regularized_incomplete_beta(df / 2, 0.5, x)
    else:
        return 0.5 * _regularized_incomplete_beta(df / 2, 0.5, x)


def t_critical(df: int, confidence: float = 0.95) -> float:
    """Get t-critical value for a confidence level and degrees of freedom.

    Finds t where P(-t < T < t) = confidence for t-distribution.
    This replaces looking up values in a t-table.

    Uses bisection method to find inverse of t_cdf.

    Args:
        df: Degrees of freedom (n - 1 for one-sample t-test)
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        T-critical value

    Raises:
        ValueError: If df ≤ 0 or confidence not in (0, 1)

    Example:
        >>> t_critical(10, 0.95)   # 95% CI with df=10
        2.228
        >>> t_critical(29, 0.95)   # 95% CI with df=29
        2.045
        >>> t_critical(1000, 0.95) # Approaches z-critical as df → ∞
        1.962
    """
    _validate_positive_int(df, "df")
    _validate_confidence(confidence)

    # Target probability for upper tail
    target = (1 + confidence) / 2

    # Bisection method to find t where t_cdf(t, df) = target
    low = 0.0
    high = 10.0

    # Expand upper bound if needed
    while t_cdf(high, df) < target:
        high *= 2

    # Bisection
    for _ in range(100):
        mid = (low + high) / 2
        if t_cdf(mid, df) < target:
            low = mid
        else:
            high = mid
        if high - low < 1e-6:
            break

    return round((low + high) / 2, 3)


def t_test_statistic(x_bar: float, mu_0: float, s: float, n: int) -> float:
    """Calculate t-test statistic for testing population mean.

    Use when: σ is UNKNOWN (use sample std s instead)

    H₀: μ = μ₀ (null hypothesis)
    H₁: μ ≠ μ₀ (two-tailed) or μ > μ₀ or μ < μ₀ (one-tailed)

    Formula: t = (x̄ - μ₀) / (s / √n)
    Degrees of freedom: df = n - 1

    Args:
        x_bar: Sample mean
        mu_0: Hypothesized population mean (null hypothesis)
        s: Sample standard deviation (not population σ)
        n: Sample size

    Returns:
        T-test statistic

    Raises:
        ValueError: If s ≤ 0 or n ≤ 0

    Example:
        >>> t_test_statistic(x_bar=105, mu_0=100, s=15, n=25)
        1.667
    """
    _validate_positive(s, "s")
    _validate_positive_int(n, "n")

    se = s / math.sqrt(n)
    return (x_bar - mu_0) / se


def p_value_from_t(t: float, df: int, alternative: str = "two-sided") -> float:
    """Calculate p-value from t-statistic.

    P-value: probability of observing a result at least as extreme
    as the one obtained, assuming H₀ is true.

    Args:
        t: T-test statistic
        df: Degrees of freedom (n - 1 for one-sample t-test)
        alternative: Type of test
            - "two-sided": H₁: μ ≠ μ₀
            - "greater": H₁: μ > μ₀
            - "less": H₁: μ < μ₀

    Returns:
        P-value

    Raises:
        ValueError: If df ≤ 0 or alternative is invalid

    Example:
        >>> p_value_from_t(2.228, 10, "two-sided")
        0.05
        >>> p_value_from_t(2.228, 10, "greater")
        0.025
    """
    _validate_positive_int(df, "df")

    valid_alternatives = ("two-sided", "greater", "less")
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}")

    if alternative == "two-sided":
        # P(|T| > |t|) = 2 * P(T > |t|)
        return 2 * (1 - t_cdf(abs(t), df))
    elif alternative == "greater":
        # P(T > t)
        return 1 - t_cdf(t, df)
    else:  # less
        # P(T < t)
        return t_cdf(t, df)


def confidence_interval_t(
    x_bar: float, s: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate confidence interval using t-distribution.

    Use when: σ is UNKNOWN (use sample std s instead)

    A 95% CI means: if we repeated sampling many times, 95% of
    intervals would contain the true population mean.

    Formula: x̄ ± t * (s / √n)
    Degrees of freedom: df = n - 1

    Args:
        x_bar: Sample mean
        s: Sample standard deviation
        n: Sample size
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Raises:
        ValueError: If s ≤ 0, n ≤ 1, or confidence invalid

    Example:
        >>> confidence_interval_t(x_bar=100, s=15, n=25, confidence=0.95)
        (93.81, 106.19)
    """
    _validate_positive(s, "s")
    if n < 2:
        raise ValueError("n must be at least 2 for t-distribution (df = n-1 >= 1)")
    _validate_confidence(confidence)

    df = n - 1
    t = t_critical(df, confidence)
    se = s / math.sqrt(n)
    me = t * se

    return (round(x_bar - me, 2), round(x_bar + me, 2))


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

    # Z-test example (σ known)
    print("--- Z-Test (σ known) ---")
    x_bar, mu_0, sigma, n = 105, 100, 15, 100
    z = z_test_statistic(x_bar, mu_0, sigma, n)
    p = p_value_from_z(z, "two-sided")
    print(f"H₀: μ = {mu_0}")
    print(f"Sample: x̄={x_bar}, σ={sigma}, n={n}")
    print(f"z-statistic: {z:.4f}")
    print(f"p-value (two-sided): {p:.4f}")
    print(f"Significant at α=0.05? {is_significant(p, 0.05)}")
    print()

    # T-test example (σ unknown)
    print("--- T-Test (σ unknown) ---")
    x_bar, mu_0, s, n = 105, 100, 15, 25
    t = t_test_statistic(x_bar, mu_0, s, n)
    df = n - 1
    p = p_value_from_t(t, df, "two-sided")
    print(f"H₀: μ = {mu_0}")
    print(f"Sample: x̄={x_bar}, s={s}, n={n}, df={df}")
    print(f"t-statistic: {t:.4f}")
    print(f"p-value (two-sided): {p:.4f}")
    print(f"Significant at α=0.05? {is_significant(p, 0.05)}")
    print()

    # T-critical values table
    print("--- T-Critical Values (replaces t-table) ---")
    print("df\\conf |  90%   |  95%   |  99%")
    print("-------|--------|--------|-------")
    for df in [5, 10, 20, 30, 60, 120]:
        t90 = t_critical(df, 0.90)
        t95 = t_critical(df, 0.95)
        t99 = t_critical(df, 0.99)
        print(f"  {df:3d}  | {t90:.3f}  | {t95:.3f}  | {t99:.3f}")
    print()

    # Confidence intervals comparison
    print("--- Confidence Intervals ---")
    x_bar, s, n = 100, 15, 25
    ci_t = confidence_interval_t(x_bar, s, n, 0.95)
    ci_z = confidence_interval(x_bar, s, n, 0.95)  # Using s as σ
    print(f"Sample: x̄={x_bar}, s={s}, n={n}")
    print(f"95% CI (t-distribution): {ci_t}")
    print(f"95% CI (z, if σ known):  {ci_z}")
    print("Note: t-interval is wider (accounts for uncertainty in s)")
