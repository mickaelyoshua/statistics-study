import pytest
import math
from stats.probability import (
    addition_rule,
    multiplication_rule,
    bayes_theorem,
    expected_value,
    binomial_coefficient,
    binomial_pmf,
    binomial_mean,
    binomial_variance,
    binomial_std,
    poisson_pmf,
    poisson_cdf,
    coin_toss,
    get_sample_means,
)


# ============ BASIC PROBABILITY RULES ============


class TestAdditionRule:
    def test_mutually_exclusive_events(self):
        # P(A or B) = P(A) + P(B) when P(A∩B) = 0
        assert addition_rule(0.3, 0.4, 0.0) == pytest.approx(0.7)

    def test_overlapping_events(self):
        # P(A or B) = P(A) + P(B) - P(A∩B)
        assert addition_rule(0.5, 0.3, 0.1) == pytest.approx(0.7)

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            addition_rule(1.5, 0.3, 0.1)

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            addition_rule(-0.1, 0.3, 0.1)


class TestMultiplicationRule:
    def test_independent_events(self):
        # P(A and B) = P(A) * P(B)
        assert multiplication_rule(0.5, 0.4) == pytest.approx(0.2)

    def test_certain_events(self):
        assert multiplication_rule(1.0, 1.0) == pytest.approx(1.0)

    def test_impossible_event(self):
        assert multiplication_rule(0.5, 0.0) == pytest.approx(0.0)

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            multiplication_rule(0.5, 1.5)


class TestBayesTheorem:
    def test_basic_calculation(self):
        # P(A|B) = P(A) * P(B|A) / P(B)
        # 0.01 * 0.9 / 0.05 = 0.18
        assert bayes_theorem(0.01, 0.05, 0.9) == pytest.approx(0.18)

    def test_equal_prior_and_posterior(self):
        # When P(B|A) = P(B), then P(A|B) = P(A)
        assert bayes_theorem(0.3, 0.5, 0.5) == pytest.approx(0.3)

    def test_zero_pb_raises(self):
        with pytest.raises(ValueError, match="pb cannot be zero"):
            bayes_theorem(0.5, 0.0, 0.5)

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            bayes_theorem(1.5, 0.5, 0.5)


class TestExpectedValue:
    def test_fair_coin(self):
        # E[X] for coin: 0.5 * 1 + 0.5 * (-1) = 0
        values = [(0.5, 1), (0.5, -1)]
        assert expected_value(values) == pytest.approx(0.0)

    def test_simple_distribution(self):
        # E[X] = 0.5 * 10 + 0.5 * 20 = 15
        values = [(0.5, 10), (0.5, 20)]
        assert expected_value(values) == pytest.approx(15.0)

    def test_weighted_distribution(self):
        values = [(0.1, 100), (0.2, 50), (0.7, 10)]
        # 0.1*100 + 0.2*50 + 0.7*10 = 10 + 10 + 7 = 27
        assert expected_value(values) == pytest.approx(27.0)

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="values cannot be empty"):
            expected_value([])

    def test_probabilities_not_sum_to_one_raises(self):
        with pytest.raises(ValueError, match="probabilities must sum to 1"):
            expected_value([(0.3, 10), (0.3, 20)])


# ============ BINOMIAL DISTRIBUTION ============


class TestBinomialCoefficient:
    def test_basic_calculation(self):
        # C(5, 2) = 10
        assert binomial_coefficient(5, 2) == 10

    def test_n_choose_zero(self):
        assert binomial_coefficient(5, 0) == 1

    def test_n_choose_n(self):
        assert binomial_coefficient(5, 5) == 1

    def test_larger_values(self):
        # C(10, 3) = 120
        assert binomial_coefficient(10, 3) == 120

    def test_k_greater_than_n_raises(self):
        with pytest.raises(ValueError, match="k cannot be greater than n"):
            binomial_coefficient(3, 5)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            binomial_coefficient(-1, 2)


class TestBinomialPMF:
    def test_coin_flip_example(self):
        # P(X=5) for 10 flips with p=0.5
        result = binomial_pmf(10, 0.5, 5)
        assert result == pytest.approx(0.24609375)

    def test_extreme_k_zero(self):
        # P(X=0) = (1-p)^n
        result = binomial_pmf(5, 0.3, 0)
        expected = 0.7 ** 5
        assert result == pytest.approx(expected)

    def test_extreme_k_equals_n(self):
        # P(X=n) = p^n
        result = binomial_pmf(5, 0.3, 5)
        expected = 0.3 ** 5
        assert result == pytest.approx(expected)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k cannot be greater than n"):
            binomial_pmf(5, 0.5, 10)

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            binomial_pmf(5, 1.5, 2)


class TestBinomialMean:
    def test_basic_calculation(self):
        # μ = n * p = 100 * 0.3 = 30
        assert binomial_mean(100, 0.3) == pytest.approx(30.0)

    def test_fair_coin(self):
        assert binomial_mean(10, 0.5) == pytest.approx(5.0)


class TestBinomialVariance:
    def test_basic_calculation(self):
        # σ² = n * p * (1-p) = 100 * 0.3 * 0.7 = 21
        assert binomial_variance(100, 0.3) == pytest.approx(21.0)


class TestBinomialStd:
    def test_basic_calculation(self):
        # σ = √(n * p * (1-p))
        result = binomial_std(100, 0.3)
        expected = math.sqrt(21.0)
        assert result == pytest.approx(expected)


# ============ POISSON DISTRIBUTION ============


class TestPoissonPMF:
    def test_basic_calculation(self):
        # P(X=5) when λ=4
        result = poisson_pmf(5, 4)
        expected = (math.exp(-4) * 4**5) / math.factorial(5)
        assert result == pytest.approx(expected)

    def test_k_zero(self):
        # P(X=0) = e^(-λ)
        result = poisson_pmf(0, 3)
        expected = math.exp(-3)
        assert result == pytest.approx(expected)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            poisson_pmf(-1, 4)

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="rate must be positive"):
            poisson_pmf(5, 0)


class TestPoissonCDF:
    def test_basic_calculation(self):
        # P(X <= 5) when λ=4
        result = poisson_cdf(5, 4)
        expected = sum(poisson_pmf(k, 4) for k in range(6))
        assert result == pytest.approx(expected)

    def test_cdf_at_zero(self):
        # P(X <= 0) = P(X=0) = e^(-λ)
        result = poisson_cdf(0, 3)
        expected = math.exp(-3)
        assert result == pytest.approx(expected)

    def test_complement(self):
        # P(X > k) = 1 - P(X <= k)
        rate = 4
        k = 5
        p_less_equal = poisson_cdf(k, rate)
        p_greater = 1 - p_less_equal
        # Should sum to 1
        assert p_less_equal + p_greater == pytest.approx(1.0)


# ============ CENTRAL LIMIT THEOREM ============


class TestCoinToss:
    def test_returns_correct_length(self):
        result = coin_toss(100)
        assert len(result) == 100

    def test_values_are_one_or_negative_one(self):
        result = coin_toss(100)
        assert all(v in [1, -1] for v in result)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            coin_toss(0)


class TestGetSampleMeans:
    def test_returns_correct_length(self):
        result = get_sample_means(50, 10, coin_toss)
        assert len(result) == 50

    def test_means_are_in_valid_range(self):
        # For coin_toss, means should be between -1 and 1
        result = get_sample_means(100, 30, coin_toss)
        assert all(-1 <= m <= 1 for m in result)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            get_sample_means(0, 10, coin_toss)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            get_sample_means(10, 0, coin_toss)


# ============ PARAMETRIZED TESTS ============


@pytest.mark.parametrize(
    "n,k,expected",
    [
        (5, 0, 1),
        (5, 1, 5),
        (5, 2, 10),
        (5, 3, 10),
        (5, 4, 5),
        (5, 5, 1),
        (10, 5, 252),
    ],
)
def test_binomial_coefficient_parametrized(n, k, expected):
    assert binomial_coefficient(n, k) == expected


@pytest.mark.parametrize(
    "n,p,expected_mean",
    [
        (10, 0.5, 5.0),
        (100, 0.3, 30.0),
        (50, 0.1, 5.0),
        (20, 0.8, 16.0),
    ],
)
def test_binomial_mean_parametrized(n, p, expected_mean):
    assert binomial_mean(n, p) == pytest.approx(expected_mean)
