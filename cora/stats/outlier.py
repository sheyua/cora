from numpy import ndarray
from pandas import DataFrame
from ..util.compare import le, gt
from ..util.constant import NAN


def is_outlier(data: DataFrame, axis: int, threshold: float) -> ndarray:
    """
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    if axis not in [0, 1]:
        raise ValueError(f'invalid axis {axis}')
    median = data.median(axis=axis)
    deviation = data.subtract(median, axis=1 - axis).abs()
    median_deviation = deviation.median(axis=axis)
    # avoid zero division
    default = median_deviation.quantile(0.01)
    assert gt(median_deviation.quantile(0.01), 0.)
    median_deviation[le(median_deviation, 0.)] = default
    assert not le(median_deviation, 0.).any()

    # this is the modified zscore
    zscore = 0.67449 * deviation.divide(median_deviation, axis=1 - axis)
    ans = gt(zscore, threshold)
    assert isinstance(ans, DataFrame)
    return ans.values


def clip_outlier(data: DataFrame, axis: int, threshold: float, inplace: bool) -> DataFrame:
    """

    """
    if inplace:
        ans = data
    else:
        ans = data.copy()
    if axis not in [0, 1]:
        raise ValueError(f'invalid axis {axis}')

    median = ans.median(axis=axis)
    deviation = ans.subtract(median, axis=1 - axis)
    median_deviation = deviation.abs().median(axis=axis)
    default = median_deviation.quantile(0.01)
    assert gt(median_deviation.quantile(0.01), 0.)
    median_deviation[le(median_deviation, 0.)] = default
    assert not le(median_deviation, 0.).any()

    # this is the modified zscore which we will clip
    ans.values[:] = 0.67449 * deviation.divide(median_deviation, axis=1 - axis)
    ans.clip(lower=-threshold, upper=threshold, inplace=True)
    ans.values[:] = ans.multiply(median_deviation, axis=1 - axis) / 0.67449
    ans.values[:] = ans.add(median, axis=1 - axis)
    assert isinstance(ans, DataFrame)
    return ans


def adjust_outlier(data: DataFrame, na_threshold: float=10., clip_threshold: float=3., inplace: bool=False) -> DataFrame:
    """
        set MAD above na_threshold to nan
        set MAD above clip_threshold to the threshold
    """
    if inplace:
        ans = data
    else:
        ans = data.copy()

    # first set extreme outliers to NAN, timeseries first then cross-sectional
    for axis in [0, 1]:
        out = is_outlier(data=ans, axis=axis, threshold=na_threshold)
        ans.values[out] = NAN

    # then clip the outlier to threshold of MAD, timeseries first then cross-sectional
    for axis in [0, 1]:
        clip_outlier(data=ans, axis=axis, threshold=clip_threshold, inplace=True)

    return ans
