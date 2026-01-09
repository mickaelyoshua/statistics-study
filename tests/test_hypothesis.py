import pytest
import math
from stats.hypothesis import (
    z_critical,
    standard_error,
    standard_error_proportion,
    margin_of_error,
    confidence_interval,
    z_test_statistic,
    p_value_from_z,
    is_significant,
    probability_within_n_std,
)


# ============ Z-CRITICAL VALUES ============


class TestZCritical:
    def test_z_critical_90(self):
        assert z_critical(0.90) == 1.645

    def test_z_critical_95(self):
        assert z_critical(0.95) == 1.960

    def test_z_critical_99(self):
        assert z_critical(0.99) == 2.576

    def test_z_critical_custom(self):
        # 80% confidence should give ~1.28
        result = z_critical(0.80)
        assert result == pytest.approx(1.28, abs=0.01)

    def test_z_critical_invalid_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            z_critical(1.5)

    def test_z_critical_zero_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            z_critical(0)


# ============ STANDARD ERROR ============


class TestStandardError:
    def test_standard_error_basic(self):
        # SE = 10 / sqrt(100) = 1.0
        assert standard_error(sigma=10, n=100) == 1.0

    def test_standard_error_large_n(self):
        # SE = 10 / sqrt(400) = 0.5
        assert standard_error(sigma=10, n=400) == 0.5

    def test_standard_error_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            standard_error(sigma=0, n=100)

    def test_standard_error_zero_n_raises(self):
        with pytest.raises(ValueError, match="n must be a positive integer"):
            standard_error(sigma=10, n=0)


class TestStandardErrorProportion:
    def test_se_proportion_basic(self):
        # SE = sqrt(0.5 * 0.5 / 100) = 0.05
        result = standard_error_proportion(p=0.5, n=100)
        assert result == pytest.approx(0.05)

    def test_se_proportion_asymmetric(self):
        # SE = sqrt(0.3 * 0.7 / 100)
        result = standard_error_proportion(p=0.3, n=100)
        expected = math.sqrt(0.3 * 0.7 / 100)
        assert result == pytest.approx(expected)

    def test_se_proportion_invalid_p_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            standard_error_proportion(p=1.5, n=100)


# ============ MARGIN OF ERROR ============


class TestMarginOfError:
    def test_margin_of_error_95(self):
        # ME = 1.96 * (10 / sqrt(100)) = 1.96
        result = margin_of_error(sigma=10, n=100, confidence=0.95)
        assert result == pytest.approx(1.96)

    def test_margin_of_error_99(self):
        # ME = 2.576 * (10 / sqrt(100)) = 2.576
        result = margin_of_error(sigma=10, n=100, confidence=0.99)
        assert result == pytest.approx(2.576)

    def test_margin_of_error_default_confidence(self):
        # Default is 95%
        result = margin_of_error(sigma=10, n=100)
        assert result == pytest.approx(1.96)


# ============ CONFIDENCE INTERVAL ============


class TestConfidenceInterval:
    def test_confidence_interval_basic(self):
        # x̄ = 100, ME = 1.96 * (15 / sqrt(100)) = 2.94
        lower, upper = confidence_interval(x_bar=100, sigma=15, n=100, confidence=0.95)
        assert lower == pytest.approx(97.06, abs=0.01)
        assert upper == pytest.approx(102.94, abs=0.01)

    def test_confidence_interval_symmetric(self):
        lower, upper = confidence_interval(x_bar=50, sigma=10, n=100, confidence=0.95)
        # Should be symmetric around mean
        assert (upper - 50) == pytest.approx(50 - lower)

    def test_confidence_interval_99_wider(self):
        lower_95, upper_95 = confidence_interval(100, 15, 100, 0.95)
        lower_99, upper_99 = confidence_interval(100, 15, 100, 0.99)
        # 99% CI should be wider
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        assert width_99 > width_95


# ============ Z-TEST STATISTIC ============


class TestZTestStatistic:
    def test_z_test_basic(self):
        # z = (105 - 100) / (15 / sqrt(100)) = 5 / 1.5 = 3.33...
        result = z_test_statistic(x_bar=105, mu_0=100, sigma=15, n=100)
        assert result == pytest.approx(3.333, abs=0.001)

    def test_z_test_at_null(self):
        # x̄ = μ₀ → z = 0
        result = z_test_statistic(x_bar=100, mu_0=100, sigma=15, n=100)
        assert result == 0.0

    def test_z_test_negative(self):
        # x̄ < μ₀ → z < 0
        result = z_test_statistic(x_bar=95, mu_0=100, sigma=15, n=100)
        assert result < 0


# ============ P-VALUE ============


class TestPValueFromZ:
    def test_p_value_two_sided_196(self):
        # z = 1.96 → p ≈ 0.05 (two-sided)
        result = p_value_from_z(1.96, "two-sided")
        assert result == pytest.approx(0.05, abs=0.001)

    def test_p_value_greater(self):
        # z = 1.96 → p ≈ 0.025 (one-sided)
        result = p_value_from_z(1.96, "greater")
        assert result == pytest.approx(0.025, abs=0.001)

    def test_p_value_less(self):
        # z = -1.96 → p ≈ 0.025 (one-sided, less)
        result = p_value_from_z(-1.96, "less")
        assert result == pytest.approx(0.025, abs=0.001)

    def test_p_value_z_zero(self):
        # z = 0 → p = 1.0 (two-sided)
        result = p_value_from_z(0, "two-sided")
        assert result == pytest.approx(1.0)

    def test_p_value_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="alternative must be one of"):
            p_value_from_z(1.96, "invalid")


# ============ IS SIGNIFICANT ============


class TestIsSignificant:
    def test_significant_below_alpha(self):
        assert is_significant(0.03, alpha=0.05) is True

    def test_not_significant_above_alpha(self):
        assert is_significant(0.10, alpha=0.05) is False

    def test_not_significant_at_alpha(self):
        # p = α → not significant (need p < α)
        assert is_significant(0.05, alpha=0.05) is False

    def test_significant_at_01(self):
        assert is_significant(0.005, alpha=0.01) is True


# ============ PROBABILITY WITHIN N STD ============


class TestProbabilityWithinNStd:
    def test_empirical_rule_1_std(self):
        lower, upper, prob = probability_within_n_std(mu=100, sigma=15, n=1)
        assert lower == 85
        assert upper == 115
        assert prob == pytest.approx(0.6827, abs=0.0001)

    def test_empirical_rule_2_std(self):
        lower, upper, prob = probability_within_n_std(mu=100, sigma=15, n=2)
        assert lower == 70
        assert upper == 130
        assert prob == pytest.approx(0.9545, abs=0.0001)

    def test_empirical_rule_3_std(self):
        lower, upper, prob = probability_within_n_std(mu=100, sigma=15, n=3)
        assert lower == 55
        assert upper == 145
        assert prob == pytest.approx(0.9973, abs=0.0001)

    def test_probability_within_n_std_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            probability_within_n_std(mu=100, sigma=0, n=1)


# ============ PARAMETRIZED TESTS ============


@pytest.mark.parametrize(
    "confidence,expected_z",
    [
        (0.90, 1.645),
        (0.95, 1.960),
        (0.99, 2.576),
    ],
)
def test_z_critical_parametrized(confidence, expected_z):
    assert z_critical(confidence) == expected_z


@pytest.mark.parametrize(
    "z,alternative,expected_p",
    [
        (0, "two-sided", 1.0),
        (1.96, "two-sided", 0.05),
        (1.96, "greater", 0.025),
        (-1.96, "less", 0.025),
        (2.576, "two-sided", 0.01),
    ],
)
def test_p_value_parametrized(z, alternative, expected_p):
    assert p_value_from_z(z, alternative) == pytest.approx(expected_p, abs=0.001)
