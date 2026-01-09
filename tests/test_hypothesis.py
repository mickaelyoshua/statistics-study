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
    t_cdf,
    t_critical,
    t_test_statistic,
    p_value_from_t,
    confidence_interval_t,
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


# ============ T-DISTRIBUTION ============


class TestTCdf:
    def test_t_cdf_zero(self):
        # t=0 always gives 0.5 (symmetric around 0)
        assert t_cdf(0, 10) == pytest.approx(0.5)

    def test_t_cdf_positive(self):
        # Known value: t=2.228, df=10 → 0.975
        assert t_cdf(2.228, 10) == pytest.approx(0.975, abs=0.001)

    def test_t_cdf_negative(self):
        # Symmetric: t_cdf(-t, df) = 1 - t_cdf(t, df)
        assert t_cdf(-2.228, 10) == pytest.approx(0.025, abs=0.001)

    def test_t_cdf_invalid_df_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            t_cdf(1.0, 0)


class TestTCritical:
    def test_t_critical_df5_95(self):
        # t-table: df=5, 95% → 2.571
        assert t_critical(5, 0.95) == pytest.approx(2.571, abs=0.01)

    def test_t_critical_df10_95(self):
        # t-table: df=10, 95% → 2.228
        assert t_critical(10, 0.95) == pytest.approx(2.228, abs=0.01)

    def test_t_critical_df30_95(self):
        # t-table: df=30, 95% → 2.042
        assert t_critical(30, 0.95) == pytest.approx(2.042, abs=0.01)

    def test_t_critical_df10_99(self):
        # t-table: df=10, 99% → 3.169
        assert t_critical(10, 0.99) == pytest.approx(3.169, abs=0.01)

    def test_t_critical_large_df_approaches_z(self):
        # As df → ∞, t approaches z
        t_large = t_critical(1000, 0.95)
        z_val = 1.96
        assert t_large == pytest.approx(z_val, abs=0.02)

    def test_t_critical_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            t_critical(10, 1.5)


class TestTTestStatistic:
    def test_t_test_basic(self):
        # t = (105 - 100) / (15 / sqrt(25)) = 5 / 3 = 1.667
        result = t_test_statistic(x_bar=105, mu_0=100, s=15, n=25)
        assert result == pytest.approx(1.667, abs=0.001)

    def test_t_test_at_null(self):
        # x̄ = μ₀ → t = 0
        result = t_test_statistic(x_bar=100, mu_0=100, s=15, n=25)
        assert result == 0.0

    def test_t_test_negative(self):
        # x̄ < μ₀ → t < 0
        result = t_test_statistic(x_bar=95, mu_0=100, s=15, n=25)
        assert result < 0

    def test_t_test_invalid_s_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            t_test_statistic(x_bar=100, mu_0=100, s=0, n=25)


class TestPValueFromT:
    def test_p_value_two_sided(self):
        # t=2.228, df=10 → p ≈ 0.05 (two-sided)
        result = p_value_from_t(2.228, 10, "two-sided")
        assert result == pytest.approx(0.05, abs=0.001)

    def test_p_value_greater(self):
        # t=2.228, df=10 → p ≈ 0.025 (one-sided)
        result = p_value_from_t(2.228, 10, "greater")
        assert result == pytest.approx(0.025, abs=0.001)

    def test_p_value_less(self):
        # t=-2.228, df=10 → p ≈ 0.025 (one-sided, less)
        result = p_value_from_t(-2.228, 10, "less")
        assert result == pytest.approx(0.025, abs=0.001)

    def test_p_value_t_zero(self):
        # t=0 → p = 1.0 (two-sided)
        result = p_value_from_t(0, 10, "two-sided")
        assert result == pytest.approx(1.0)

    def test_p_value_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="alternative must be one of"):
            p_value_from_t(1.0, 10, "invalid")


class TestConfidenceIntervalT:
    def test_ci_t_basic(self):
        # x̄=100, s=15, n=25, df=24, t≈2.064
        # ME = 2.064 * (15/5) = 6.19
        lower, upper = confidence_interval_t(x_bar=100, s=15, n=25, confidence=0.95)
        assert lower == pytest.approx(93.81, abs=0.1)
        assert upper == pytest.approx(106.19, abs=0.1)

    def test_ci_t_symmetric(self):
        lower, upper = confidence_interval_t(x_bar=50, s=10, n=16, confidence=0.95)
        # Should be symmetric around mean
        assert (upper - 50) == pytest.approx(50 - lower, abs=0.01)

    def test_ci_t_99_wider(self):
        lower_95, upper_95 = confidence_interval_t(100, 15, 25, 0.95)
        lower_99, upper_99 = confidence_interval_t(100, 15, 25, 0.99)
        # 99% CI should be wider
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        assert width_99 > width_95

    def test_ci_t_small_n_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            confidence_interval_t(x_bar=100, s=15, n=1, confidence=0.95)


# ============ PARAMETRIZED T-DISTRIBUTION TESTS ============


@pytest.mark.parametrize(
    "df,confidence,expected_t",
    [
        (5, 0.90, 2.015),
        (5, 0.95, 2.571),
        (10, 0.95, 2.228),
        (10, 0.99, 3.169),
        (30, 0.95, 2.042),
    ],
)
def test_t_critical_parametrized(df, confidence, expected_t):
    assert t_critical(df, confidence) == pytest.approx(expected_t, abs=0.01)


@pytest.mark.parametrize(
    "t,df,alternative,expected_p",
    [
        (0, 10, "two-sided", 1.0),
        (2.228, 10, "two-sided", 0.05),
        (2.228, 10, "greater", 0.025),
        (-2.228, 10, "less", 0.025),
        (3.169, 10, "two-sided", 0.01),
    ],
)
def test_p_value_from_t_parametrized(t, df, alternative, expected_p):
    assert p_value_from_t(t, df, alternative) == pytest.approx(expected_p, abs=0.002)
