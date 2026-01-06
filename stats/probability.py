"""Probability rules, distributions, and expected value calculations.

Probability Rules:
    addition_rule: P(A or B) = P(A) + P(B) - P(A and B)
    multiplication_rule: P(A and B) = P(A) * P(B) or P(A) * P(B|A)
    bayes_theorem: P(A|B) = P(A) * P(B|A) / P(B)
    expected_value: E[X] = sum of probability * value

Binomial Distribution:
    binomial_coefficient: C(n,k) = n! / (k! * (n-k)!)
    binomial_pmf: P(X=k) for binomial distribution
    binomial_mean: μ = n * p
    binomial_variance: σ² = n * p * (1-p)
    binomial_std: σ = sqrt(variance)

Poisson Distribution:
    poisson_pmf: P(X=k) for Poisson distribution
    poisson_cdf: P(X <= k) for Poisson distribution

Central Limit Theorem:
    coin_toss: Generate random coin tosses
    get_sample_means: Demonstrate CLT with sample means
    print_histogram: Display histogram in terminal

All probabilities must be in range [0, 1].
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
import plotext as plt


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_probability(p: float, name: str) -> None:
    """Validate that a probability is in [0, 1]."""
    if not 0 <= p <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_positive_int(n: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_int(k: int, name: str) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(k, int) or k < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


# =============================================================================
# Basic Probability Rules
# =============================================================================


def addition_rule(pa: float, pb: float, pab: float) -> float:
    """Calculate P(A or B) using the addition rule.

    Formula: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

    Args:
        pa: Probability of event A
        pb: Probability of event B
        pab: Probability of both A and B occurring

    Returns:
        Probability of A or B occurring

    Raises:
        ValueError: If any probability is not in [0, 1]

    Example:
        >>> addition_rule(0.5, 0.3, 0.1)  # P(A)=0.5, P(B)=0.3, P(A∩B)=0.1
        0.7
    """
    _validate_probability(pa, "pa")
    _validate_probability(pb, "pb")
    _validate_probability(pab, "pab")
    return pa + pb - pab


def multiplication_rule(p1: float, p2: float) -> float:
    """Calculate P(A and B) using the multiplication rule.

    For independent events: P(A ∩ B) = P(A) * P(B)
    For dependent events: P(A ∩ B) = P(A) * P(B|A)

    Args:
        p1: P(A) - probability of first event
        p2: P(B) for independent events, or P(B|A) for dependent events

    Returns:
        Probability of both events occurring

    Raises:
        ValueError: If any probability is not in [0, 1]

    Example:
        >>> multiplication_rule(0.5, 0.4)  # Independent events
        0.2
    """
    _validate_probability(p1, "p1")
    _validate_probability(p2, "p2")
    return p1 * p2


def bayes_theorem(pa: float, pb: float, pba: float) -> float:
    """Calculate P(A|B) using Bayes' theorem.

    Formula: P(A|B) = P(A) * P(B|A) / P(B)

    Used to update prior probability with new evidence.

    Args:
        pa: P(A) - prior probability of A
        pb: P(B) - probability of evidence B
        pba: P(B|A) - likelihood of B given A

    Returns:
        P(A|B) - posterior probability of A given B

    Raises:
        ValueError: If any probability is not in [0, 1]
        ValueError: If pb is zero (division by zero)

    Example:
        >>> bayes_theorem(0.01, 0.05, 0.9)  # Disease test
        0.18
    """
    _validate_probability(pa, "pa")
    _validate_probability(pb, "pb")
    _validate_probability(pba, "pba")
    if pb == 0:
        raise ValueError("pb cannot be zero")
    return (pa * pba) / pb


def expected_value(values: Sequence[tuple[float, float | int]]) -> float:
    """Calculate expected value E[X] = Σ(p * x).

    Args:
        values: Sequence of (probability, value) tuples

    Returns:
        Expected value of the distribution

    Raises:
        ValueError: If values is empty
        ValueError: If any probability is not in [0, 1]
        ValueError: If probabilities don't sum to 1

    Example:
        >>> expected_value([(0.5, 10), (0.5, 20)])
        15.0
    """
    if len(values) == 0:
        raise ValueError("values cannot be empty")

    total_prob = 0.0
    for p, _ in values:
        _validate_probability(p, "probability")
        total_prob += p

    if abs(total_prob - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1")

    return sum(p * x for p, x in values)


# =============================================================================
# Binomial Distribution
# Discrete distribution for number of successes in n independent trials
# Assumptions: Fixed n trials, two outcomes, constant p, independent trials
# =============================================================================


def binomial_coefficient(n: int, k: int) -> int:
    """Calculate binomial coefficient C(n,k) = n! / (k! * (n-k)!).

    Also known as "n choose k" - the number of ways to choose k items from n.

    Args:
        n: Total number of items
        k: Number of items to choose

    Returns:
        Number of combinations

    Raises:
        ValueError: If n < 0 or k < 0
        ValueError: If k > n

    Example:
        >>> binomial_coefficient(5, 2)
        10
    """
    _validate_non_negative_int(n, "n")
    _validate_non_negative_int(k, "k")
    if k > n:
        raise ValueError("k cannot be greater than n")
    return math.comb(n, k)


def binomial_pmf(n: int, p: float, k: int) -> float:
    """Calculate P(X=k) for binomial distribution.

    Formula: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)

    Args:
        n: Number of trials
        p: Probability of success on each trial
        k: Number of successes

    Returns:
        Probability of exactly k successes

    Raises:
        ValueError: If n <= 0 or k < 0 or k > n
        ValueError: If p is not in [0, 1]

    Example:
        >>> binomial_pmf(10, 0.5, 5)  # 5 heads in 10 coin flips
        0.24609375
    """
    _validate_positive_int(n, "n")
    _validate_non_negative_int(k, "k")
    _validate_probability(p, "p")
    if k > n:
        raise ValueError("k cannot be greater than n")
    return binomial_coefficient(n, k) * (p**k) * ((1 - p) ** (n - k))


def binomial_mean(n: int, p: float) -> float:
    """Calculate mean (expected value) of binomial distribution.

    Formula: μ = n * p

    Args:
        n: Number of trials
        p: Probability of success

    Returns:
        Expected number of successes

    Example:
        >>> binomial_mean(100, 0.3)
        30.0
    """
    _validate_positive_int(n, "n")
    _validate_probability(p, "p")
    return n * p


def binomial_variance(n: int, p: float) -> float:
    """Calculate variance of binomial distribution.

    Formula: σ² = n * p * (1-p)

    Args:
        n: Number of trials
        p: Probability of success

    Returns:
        Variance of the distribution

    Example:
        >>> binomial_variance(100, 0.3)
        21.0
    """
    _validate_positive_int(n, "n")
    _validate_probability(p, "p")
    return n * p * (1 - p)


def binomial_std(n: int, p: float) -> float:
    """Calculate standard deviation of binomial distribution.

    Formula: σ = sqrt(n * p * (1-p))

    Args:
        n: Number of trials
        p: Probability of success

    Returns:
        Standard deviation of the distribution

    Example:
        >>> binomial_std(100, 0.3)
        4.58257569...
    """
    return math.sqrt(binomial_variance(n, p))


# =============================================================================
# Poisson Distribution
# Discrete distribution for number of events in a fixed interval
# Assumptions: Constant rate, independent events
# Example: Number of customers entering a store per hour
# =============================================================================


def poisson_pmf(k: int, rate: float) -> float:
    """Calculate P(X=k) for Poisson distribution.

    Formula: P(X=k) = (e^(-λ) * λ^k) / k!

    Args:
        k: Number of events (must be >= 0)
        rate: λ (lambda) - expected number of events (must be > 0)

    Returns:
        Probability of exactly k events

    Raises:
        ValueError: If k < 0 or rate <= 0

    Example:
        >>> poisson_pmf(5, 4)  # 5 events when average is 4
        0.15629...
    """
    _validate_non_negative_int(k, "k")
    _validate_positive(rate, "rate")
    return (math.exp(-rate) * rate**k) / math.factorial(k)


def poisson_cdf(k: int, rate: float) -> float:
    """Calculate P(X <= k) for Poisson distribution.

    Formula: P(X <= k) = Σ P(X=i) for i from 0 to k

    Args:
        k: Maximum number of events
        rate: λ (lambda) - expected number of events

    Returns:
        Probability of at most k events

    Raises:
        ValueError: If k < 0 or rate <= 0

    Example:
        >>> poisson_cdf(5, 4)  # At most 5 events when average is 4
        0.78513...
    """
    _validate_non_negative_int(k, "k")
    _validate_positive(rate, "rate")
    return sum(poisson_pmf(i, rate) for i in range(k + 1))


# =============================================================================
# Central Limit Theorem Demonstration
# The distribution of sample means approaches normal distribution
# regardless of the original distribution
# =============================================================================


def coin_toss(n: int) -> np.ndarray:
    """Generate n random coin tosses (1 or -1).

    Args:
        n: Number of coin tosses

    Returns:
        Array of n values, each 1 or -1

    Raises:
        ValueError: If n <= 0

    Example:
        >>> coin_toss(5)  # Returns array like [1, -1, 1, 1, -1]
    """
    _validate_positive_int(n, "n")
    return np.random.choice([1, -1], size=n)


def get_sample_means(
    k: int, n: int, sample_generator: Callable[[int], np.ndarray | Sequence]
) -> list[float]:
    """Generate k samples of size n and return their means.

    Used to demonstrate the Central Limit Theorem.

    Args:
        k: Number of samples to generate
        n: Size of each sample
        sample_generator: Function that takes n and returns a sample

    Returns:
        List of k sample means

    Raises:
        ValueError: If k <= 0 or n <= 0

    Example:
        >>> means = get_sample_means(1000, 30, coin_toss)
        >>> len(means)
        1000
    """
    _validate_positive_int(k, "k")
    _validate_positive_int(n, "n")
    return [float(np.mean(sample_generator(n))) for _ in range(k)]


def print_histogram(sample_means: Sequence[float]) -> None:
    """Display histogram of sample means in terminal.

    Args:
        sample_means: Sequence of sample means to plot
    """
    plt.hist(list(sample_means), bins=30)
    plt.show()


# =============================================================================
# Main - Examples
# =============================================================================


if __name__ == "__main__":
    # Poisson distribution examples
    print("=== Poisson Distribution ===")
    print(f"P(X=5) when λ=4: {poisson_pmf(5, 4):.4f}")
    print(f"P(X≤5) when λ=4: {poisson_cdf(5, 4):.4f}")
    print()

    print(f"P(X=12) when λ=14: {poisson_pmf(12, 14):.4f}")
    print(f"P(X≤12) when λ=14: {poisson_cdf(12, 14):.4f}")
    print(f"P(X>12) when λ=14: {1 - poisson_cdf(12, 14):.4f}")
