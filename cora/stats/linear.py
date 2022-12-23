from typing import Tuple
from numpy import matmul
from pandas import DataFrame, Series
from ..util.compare import le
from .covariance import invert, max_entropy


def decompose_label(feature: DataFrame, label: Series, weight: Series) -> Tuple[Series, Series]:
    """

    """
    # now check that factor are non-zero on any given day
    remove = list()
    for col in feature.columns:
        leverage = feature[col].abs().sum()
        if le(leverage, 0.):
            remove.append(col)
    if len(remove):
        feature = feature.drop(columns=remove)

    # (1) align index
    index = feature.index.intersection(label.index)
    index = index.intersection(weight.index).sort_values()
    feature = feature.reindex(index=index)
    label = label.reindex(index=index)
    weight = weight.reindex(index=index)
    assert weight.isnull().sum() == 0
    assert feature.isnull().sum(axis=0).sum() == 0
    label.fillna(value=0., inplace=True)

    # (2) compute var-covar X.T w X, left is X.T w
    left = feature.values.T * weight.values.reshape([1, -1])
    covar = matmul(left, feature.values)
    # this need shrinkage
    covar = max_entropy(data=covar, prior=None)
    kernel = invert(arr=covar)

    # (3) compute the regression coef_ X.T w y and resid
    right = matmul(left, label.values)
    beta = matmul(kernel, right)
    resid = label.values - matmul(feature.values, beta)

    # package back to series
    beta = Series(data=beta, index=feature.columns, name='beta')
    resid = Series(data=resid, index=feature.index, name='resid')
    if len(remove):
        index = beta.index.tolist() + remove
        beta = beta.reindex(index=index, fill_value=0.)
    return beta, resid
