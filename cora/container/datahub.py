from typing import List, Tuple, Dict
from numpy import ndarray, concatenate, sqrt, diag, matmul, nan_to_num, nansum, power
from pandas import DataFrame, Series
from ..stats.covariance import invert, max_entropy
from ..util.compare import gt, ge


class DataHub(object):
    """
        erase asset and date information, use ndarray to save ram
    """
    def __init__(self, feature_column: List[str], label_name: str) -> None:
        """

        """
        self.feature_column = feature_column
        self.label_name = label_name
        self._feature = list()
        self._weight = list()
        self._label = list()
        self._specific = list()

    def append(self, feature: DataFrame, label: Series, weight: Series, specific: Series) -> None:
        """

        """
        for col in self.feature_column:
            if col not in feature.columns:
                raise KeyError(f'{col} feature not found')
        if label.name != self.label_name:
            raise ValueError(f'label name mismatch')
        if weight.isnull().sum() + label.isnull().sum() + specific.isnull().sum():
            raise ValueError(f'only feature can be nan')
        # align index
        index = label.index.intersection(feature.index).intersection(specific.index)
        index = index.intersection(weight.index).sort_values()
        # adjust each data
        label = label.reindex(index=index)
        feature = feature.reindex(index=index, columns=self.feature_column)
        weight = weight.reindex(index=index)
        specific = specific.reindex(index=index)
        # add to the list
        self._feature.append(feature.values)
        self._label.append(label.values)
        self._weight.append(weight.values)
        self._specific.append(specific.values)

    @property
    def num_feature(self) -> int: return len(self.feature_column)

    @property
    def feature(self) -> ndarray:
        """

        """
        if len(self._feature) == 1:
            ans, *_ = self._feature
        else:
            ans = concatenate(self._feature, axis=0)
            self._feature = [ans]
        return ans

    @property
    def label(self) -> ndarray:
        """

        """
        if len(self._label) == 1:
            ans, *_ = self._label
        else:
            ans = concatenate(self._label, axis=0)
            self._label = [ans]
        return ans

    @property
    def weight(self) -> ndarray:
        """

        """
        if len(self._weight) == 1:
            ans, *_ = self._weight
        else:
            ans = concatenate(self._weight, axis=0)
            self._weight = [ans]
        return ans

    @property
    def specific(self) -> ndarray:
        """

        """
        if len(self._specific) == 1:
            ans, *_ = self._specific
        else:
            ans = concatenate(self._specific, axis=0)
            self._specific = [ans]
        return ans

    def compute_linear(self, shrinkage: float, prior: Dict[str, float]) -> DataFrame:
        """
            the proper way to run would be to
            (1) specify prior
            (2) cross validate between different shrinkage

            here we simply use one shrinkage to prior
        """
        x_data = self.feature.copy()
        nan_to_num(x_data, copy=False, nan=0.)
        weight = self.weight / self.weight.sum()
        # compute X.T w X first
        left = x_data * weight.reshape([-1, 1])
        x_var = matmul(left.T, x_data)
        x_var = max_entropy(data=x_var, prior=None)
        scale = sqrt(diag(x_var))
        assert gt(scale, 0.).all()
        # now compute X.T w y
        covar = matmul(left.T, self.label)
        kernel = invert(x_var)
        beta = matmul(kernel.T, covar)

        # now get the prior by scale base coef_ up
        prior = Series(prior, name='prior').reindex(index=self.feature_column, fill_value=0.)
        assert ge(prior.abs().mean(), 0.)
        if gt(prior.abs().mean(), 0.):
            prior = prior / prior.abs().mean()
        prior = (beta * scale).mean() * prior.values / scale

        # now we shrink towards this prior, however this coef needs to be adjusted
        coef_ = prior * shrinkage + (1 - shrinkage) * beta
        x_one = matmul(x_data, coef_)
        adjust = (x_one * self.weight * self.label).sum() / (x_one * self.weight * x_one).sum()
        coef_ = coef_ * adjust

        # final ans
        ans = dict(scale=scale, beta=beta, prior=prior, coef_=coef_)
        ans = DataFrame.from_dict(ans, orient='columns')
        ans.index = self.feature_column
        ans.index.name = 'feature'
        return ans

    def exfoliate_non_linear(self, linear: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """
            remove the linear component
            scale both feature and label by spec risk
            scale weights by 1 / variance
        """
        # first take out the linear components
        x_data = self.feature
        y_pred = x_data * linear.reshape([1, -1])
        y_pred = nansum(y_pred, axis=1)
        y_data = self.label - y_pred
        # then scale the problem, first scale the inputs and output
        assert gt(self.specific.min(), 0.)
        x_data = x_data / self.specific.reshape([-1, 1])
        y_data = y_data / self.specific
        # then scale the weight to put variance back
        weight = self.weight * power(self.specific, 2)
        return x_data, y_data, weight
