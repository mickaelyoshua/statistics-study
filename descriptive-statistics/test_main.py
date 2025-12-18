import pytest
from main import (
    mean,
    median,
    mode,
    variance,
    std,
    sample_variance,
    sample_std,
    skewness,
    arr_range,
    iqr,
    standardized_statistical_moment,
)


# ============ CENTRAL TENDENCIES ============


class TestMean:
    def test_mean_basic(self):
        # Arrange
        arr = [1, 2, 3, 4, 5]
        # Act
        result = mean(arr)
        # Assert
        assert result == 3.0

    def test_mean_single_element(self):
        assert mean([42]) == 42.0

    def test_mean_with_floats(self):
        arr = [1.5, 2.5, 3.5]
        assert mean(arr) == pytest.approx(2.5)

    def test_mean_with_negatives(self):
        arr = [-2, -1, 0, 1, 2]
        assert mean(arr) == 0.0

    def test_mean_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            mean([])


class TestMedian:
    def test_median_odd_length(self):
        # Arrange
        arr = [3, 1, 2]  # sorted: [1, 2, 3]
        # Act & Assert
        assert median(arr) == 2

    def test_median_even_length(self):
        arr = [4, 1, 3, 2]  # sorted: [1, 2, 3, 4]
        assert median(arr) == 2.5  # (2 + 3) / 2

    def test_median_single_element(self):
        assert median([7]) == 7

    def test_median_unsorted_input(self):
        arr = [9, 1, 5, 3, 7]  # sorted: [1, 3, 5, 7, 9]
        assert median(arr) == 5

    def test_median_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            median([])


class TestMode:
    def test_mode_single_mode(self):
        arr = [1, 2, 2, 3]
        result = mode(arr)
        assert result == [2]

    def test_mode_multiple_modes(self):
        arr = [1, 1, 2, 2, 3]
        result = mode(arr)
        assert set(result) == {1, 2}

    def test_mode_all_same(self):
        arr = [5, 5, 5, 5]
        result = mode(arr)
        assert result == [5]

    def test_mode_no_mode_raises(self):
        arr = [1, 2, 3, 4]  # all unique
        with pytest.raises(ValueError, match="No mode"):
            mode(arr)

    def test_mode_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            mode([])


# ============ DISPERSION ============


class TestVariance:
    def test_variance_basic(self):
        # Population variance of [1,2,3,4,5]
        # mean = 3, deviations = [-2,-1,0,1,2], squared = [4,1,0,1,4], sum = 10
        # variance = 10/5 = 2.0
        arr = [1, 2, 3, 4, 5]
        assert variance(arr) == 2.0

    def test_variance_constant_values(self):
        arr = [5, 5, 5, 5]
        assert variance(arr) == 0.0

    def test_variance_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            variance([])


class TestStd:
    def test_std_basic(self):
        arr = [1, 2, 3, 4, 5]
        # std = sqrt(2) ≈ 1.414
        assert std(arr) == pytest.approx(2**0.5)

    def test_std_constant_values(self):
        arr = [3, 3, 3]
        assert std(arr) == 0.0

    def test_std_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            std([])


class TestSampleVariance:
    def test_sample_variance_basic(self):
        # Sample variance uses n-1 (Bessel's correction)
        # For [1,2,3,4,5]: sum of squared deviations = 10
        # sample_variance = 10/4 = 2.5
        arr = [1, 2, 3, 4, 5]
        assert sample_variance(arr) == 2.5

    def test_sample_variance_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            sample_variance([])


class TestSampleStd:
    def test_sample_std_basic(self):
        arr = [1, 2, 3, 4, 5]
        # sample_std = sqrt(2.5) ≈ 1.581
        assert sample_std(arr) == pytest.approx(2.5**0.5)

    def test_sample_std_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            sample_std([])


class TestSkewness:
    def test_skewness_positive_right_skewed(self):
        # Right-skewed: long tail to the right
        arr = [1, 1, 1, 1, 2, 2, 2, 3, 3, 10]
        result = skewness(arr)
        assert result > 0

    def test_skewness_negative_left_skewed(self):
        # Left-skewed: long tail to the left
        arr = [1, 8, 8, 9, 9, 9, 10, 10, 10, 10]
        result = skewness(arr)
        assert result < 0

    def test_skewness_symmetric_near_zero(self):
        # Symmetric distribution
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        result = skewness(arr)
        assert result == pytest.approx(0, abs=0.1)

    def test_skewness_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            skewness([])


class TestRange:
    def test_range_basic(self):
        arr = [3, 1, 7, 2, 9]
        assert arr_range(arr) == 8  # 9 - 1

    def test_range_single_element(self):
        assert arr_range([5]) == 0

    def test_range_same_values(self):
        arr = [4, 4, 4]
        assert arr_range(arr) == 0

    def test_range_with_negatives(self):
        arr = [-5, 0, 5]
        assert arr_range(arr) == 10

    def test_range_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            arr_range([])


class TestIQR:
    def test_iqr_odd_length(self):
        # arr3 from main.py: [2,2,3,4,5,5,5,6,6,6,6,7,7] (n=13)
        # Q1 = median([2,2,3,4,5,5]) = 3.5
        # Q3 = median([6,6,6,6,7,7]) = 6
        # IQR = 6 - 3.5 = 2.5
        arr = [2, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7]
        assert iqr(arr) == pytest.approx(2.5)

    def test_iqr_even_length(self):
        # arr4 from main.py: [1,1,2,2,2,2,3,3,3,3,4,4,4,6] (n=14)
        # Lower half: [1,1,2,2,2,2,3] -> Q1 = 2
        # Upper half: [3,3,3,4,4,4,6] -> Q3 = 4
        # IQR = 4 - 2 = 2
        arr = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6]
        assert iqr(arr) == pytest.approx(2.0)

    def test_iqr_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            iqr([])


# ============ STANDARDIZED MOMENTS ============


class TestStandardizedMoment:
    def test_moment_zero_returns_one(self):
        arr = [1, 2, 3, 4, 5]
        assert standardized_statistical_moment(arr, 0) == 1.0

    def test_moment_one_returns_zero(self):
        arr = [1, 2, 3, 4, 5]
        assert standardized_statistical_moment(arr, 1) == 0.0

    def test_moment_two_returns_one(self):
        arr = [1, 2, 3, 4, 5]
        assert standardized_statistical_moment(arr, 2) == 1.0

    def test_moment_three_equals_skewness(self):
        arr = [1, 2, 2, 3, 3, 3, 4, 4, 5]
        moment_3 = standardized_statistical_moment(arr, 3)
        skew = skewness(arr)
        assert moment_3 == pytest.approx(skew)

    def test_moment_negative_raises(self):
        arr = [1, 2, 3]
        with pytest.raises(ValueError, match="less then 0"):
            standardized_statistical_moment(arr, -1)

    def test_moment_empty_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            standardized_statistical_moment([], 2)


# ============ PARAMETRIZED TESTS ============


@pytest.mark.parametrize(
    "arr,expected",
    [
        ([1, 2, 3], 2.0),
        ([10, 20, 30, 40], 25.0),
        ([0, 0, 0], 0.0),
        ([-1, 1], 0.0),
        ([2.5, 3.5, 4.0], pytest.approx(10 / 3)),
    ],
)
def test_mean_parametrized(arr, expected):
    assert mean(arr) == expected


@pytest.mark.parametrize(
    "arr,expected",
    [
        ([1, 2, 3], 2),
        ([1, 2, 3, 4], 2.5),
        ([5], 5),
        ([1, 100], 50.5),
    ],
)
def test_median_parametrized(arr, expected):
    assert median(arr) == expected
