import pytest
import math
from stats.distributions import (
    density_normal_distribution,
    z_score,
    get_x_from_z_score,
    standard_normal_cdf,
    normal_cdf,
)


# ============ Z-SCORE ============


class TestZScore:
    def test_z_score_at_mean(self):
        # At the mean, z-score should be 0
        assert z_score(sigma=10, mu=50, x=50) == 0.0

    def test_z_score_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            z_score(sigma=0, mu=50, x=50)

    def test_z_score_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            z_score(sigma=-10, mu=50, x=50)

    def test_z_score_one_std_above(self):
        # One std deviation above mean -> z = 1
        assert z_score(sigma=10, mu=50, x=60) == 1.0

    def test_z_score_one_std_below(self):
        # One std deviation below mean -> z = -1
        assert z_score(sigma=10, mu=50, x=40) == -1.0

    def test_z_score_two_std_above(self):
        assert z_score(sigma=15, mu=100, x=130) == 2.0

    def test_z_score_fractional(self):
        # (75 - 50) / 10 = 2.5
        assert z_score(sigma=10, mu=50, x=75) == 2.5


class TestGetXFromZScore:
    def test_z_zero_returns_mean(self):
        assert get_x_from_z_score(sigma=10, mu=50, z=0) == 50

    def test_get_x_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            get_x_from_z_score(sigma=0, mu=50, z=1)

    def test_get_x_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            get_x_from_z_score(sigma=-10, mu=50, z=1)

    def test_z_one_returns_mean_plus_sigma(self):
        assert get_x_from_z_score(sigma=10, mu=50, z=1) == 60

    def test_z_negative_one_returns_mean_minus_sigma(self):
        assert get_x_from_z_score(sigma=10, mu=50, z=-1) == 40

    def test_roundtrip_z_score(self):
        # z_score and get_x_from_z_score should be inverses
        sigma, mu, x = 15, 100, 121
        z = z_score(sigma, mu, x)
        result = get_x_from_z_score(sigma, mu, z)
        assert result == pytest.approx(x)


# ============ PROBABILITY DENSITY FUNCTION ============


class TestDensityNormalDistribution:
    def test_pdf_at_mean_standard_normal(self):
        # For standard normal (mu=0, sigma=1), PDF at x=0 is 1/sqrt(2*pi)
        expected = 1 / math.sqrt(2 * math.pi)
        result = density_normal_distribution(sigma=1, mu=0, x=0)
        assert result == pytest.approx(expected)

    def test_pdf_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            density_normal_distribution(sigma=0, mu=0, x=0)

    def test_pdf_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            density_normal_distribution(sigma=-1, mu=0, x=0)

    def test_pdf_symmetric_around_mean(self):
        # PDF should be symmetric: f(mu + a) == f(mu - a)
        sigma, mu = 10, 50
        result_above = density_normal_distribution(sigma, mu, x=60)
        result_below = density_normal_distribution(sigma, mu, x=40)
        assert result_above == pytest.approx(result_below)

    def test_pdf_decreases_away_from_mean(self):
        # PDF should be highest at mean and decrease as we move away
        sigma, mu = 10, 50
        at_mean = density_normal_distribution(sigma, mu, x=50)
        one_std = density_normal_distribution(sigma, mu, x=60)
        two_std = density_normal_distribution(sigma, mu, x=70)
        assert at_mean > one_std > two_std

    def test_pdf_positive(self):
        # PDF should always be positive
        result = density_normal_distribution(sigma=5, mu=20, x=35)
        assert result > 0


# ============ CUMULATIVE DISTRIBUTION FUNCTION ============


class TestStandardNormalCDF:
    def test_cdf_at_zero(self):
        # P(Z <= 0) = 0.5 for standard normal
        assert standard_normal_cdf(0) == pytest.approx(0.5)

    def test_cdf_at_z_one(self):
        # P(Z <= 1) ≈ 0.8413 (from z-table)
        assert standard_normal_cdf(1) == pytest.approx(0.8413, abs=0.0001)

    def test_cdf_at_z_two(self):
        # P(Z <= 2) ≈ 0.9772
        assert standard_normal_cdf(2) == pytest.approx(0.9772, abs=0.0001)

    def test_cdf_at_z_negative_one(self):
        # P(Z <= -1) ≈ 0.1587
        assert standard_normal_cdf(-1) == pytest.approx(0.1587, abs=0.0001)

    def test_cdf_symmetry(self):
        # P(Z <= -a) = 1 - P(Z <= a)
        z = 1.5
        assert standard_normal_cdf(-z) == pytest.approx(1 - standard_normal_cdf(z))

    def test_cdf_large_positive_z(self):
        # P(Z <= large) approaches 1
        assert standard_normal_cdf(5) == pytest.approx(1.0, abs=0.0001)

    def test_cdf_large_negative_z(self):
        # P(Z <= -large) approaches 0
        assert standard_normal_cdf(-5) == pytest.approx(0.0, abs=0.0001)


class TestNormalCDF:
    def test_cdf_at_mean(self):
        # P(X <= mu) = 0.5
        assert normal_cdf(sigma=10, mu=50, x=50) == pytest.approx(0.5)

    def test_cdf_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            normal_cdf(sigma=0, mu=50, x=50)

    def test_cdf_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            normal_cdf(sigma=-10, mu=50, x=50)

    def test_cdf_one_std_above(self):
        # P(X <= mu + sigma) ≈ 0.8413
        assert normal_cdf(sigma=10, mu=50, x=60) == pytest.approx(0.8413, abs=0.0001)

    def test_cdf_one_std_below(self):
        # P(X <= mu - sigma) ≈ 0.1587
        assert normal_cdf(sigma=10, mu=50, x=40) == pytest.approx(0.1587, abs=0.0001)

    def test_cdf_iq_example(self):
        # IQ: mu=100, sigma=15
        # P(IQ <= 115) = P(Z <= 1) ≈ 0.8413
        result = normal_cdf(sigma=15, mu=100, x=115)
        assert result == pytest.approx(0.8413, abs=0.0001)

    def test_cdf_range_probability(self):
        # P(85 < X <= 115) for IQ should be about 68% (empirical rule)
        sigma, mu = 15, 100
        prob = normal_cdf(sigma, mu, 115) - normal_cdf(sigma, mu, 85)
        assert prob == pytest.approx(0.6827, abs=0.001)


# ============ PARAMETRIZED TESTS ============


@pytest.mark.parametrize(
    "sigma,mu,x,expected_z",
    [
        (10, 50, 50, 0.0),
        (10, 50, 60, 1.0),
        (10, 50, 40, -1.0),
        (15, 100, 130, 2.0),
        (2.5, 27, 29.5, 1.0),
    ],
)
def test_z_score_parametrized(sigma, mu, x, expected_z):
    assert z_score(sigma, mu, x) == pytest.approx(expected_z)


@pytest.mark.parametrize(
    "z,expected_cdf",
    [
        (0, 0.5),
        (1, 0.8413),
        (-1, 0.1587),
        (2, 0.9772),
        (-2, 0.0228),
        (3, 0.9987),
    ],
)
def test_standard_normal_cdf_parametrized(z, expected_cdf):
    assert standard_normal_cdf(z) == pytest.approx(expected_cdf, abs=0.0001)
