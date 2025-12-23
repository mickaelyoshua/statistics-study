"""Probability rules and expected value calculations.

Functions:
    addition_rule: P(A or B) = P(A) + P(B) - P(A and B)
    multiplication_rule: P(A and B) = P(A) * P(B) or P(A) * P(B|A)
    bayes_theorem: P(A|B) = P(A) * P(B|A) / P(B)
    expected_value: E[X] = sum of probability * value

All probabilities must be in range [0, 1].
"""
from collections.abc import Sequence


def _validate_probability(p: float, name: str) -> None:
    if not 0 <= p <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def addition_rule(pa: float, pb: float, pab: float) -> float:
    """P(A or B) = P(A) + P(B) - P(A and B). For mutually exclusive events, pab=0."""
    _validate_probability(pa, "pa")
    _validate_probability(pb, "pb")
    _validate_probability(pab, "pab")
    return pa + pb - pab


def multiplication_rule(p1: float, p2: float) -> float:
    """P(A and B). Pass P(B) for independent events, P(B|A) for dependent."""
    _validate_probability(p1, "p1")
    _validate_probability(p2, "p2")
    return p1 * p2


def bayes_theorem(pa: float, pb: float, pba: float) -> float:
    """P(A|B) = P(A) * P(B|A) / P(B). Updates prior probability with new evidence."""
    _validate_probability(pa, "pa")
    _validate_probability(pb, "pb")
    _validate_probability(pba, "pba")
    if pb == 0:
        raise ValueError("pb cannot be zero")
    return (pa * pba) / pb


def expected_value(values: Sequence[tuple[float, float | int]]) -> float:
    """E[X] = sum(p * x). Probabilities must sum to 1."""
    if len(values) == 0:
        raise ValueError("values cannot be empty")

    total_prob = 0.0
    for p, x in values:
        _validate_probability(p, "probability")
        total_prob += p

    if abs(total_prob - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1")

    return sum([p * x for p, x in values])


if __name__ == "__main__":
    print(multiplication_rule(7 / 10, 6 / 9))
    print(bayes_theorem(1 / 2, 5 / 12, 1 / 2))
