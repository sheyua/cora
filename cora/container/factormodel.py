from typing import Optional
from numpy import allclose, ndarray, power, matmul
from pandas import DataFrame, Series, Index
from ..stats.covariance import invert
from ..util.compare import EPSILON


class FactorModel(object):
    """

    """
    @property
    def time_unit(self) -> str: return 'annual'

    @property
    def risk_unit(self) -> str: return 'decimal'

    @property
    def num_asset(self) -> int: return len(self.exposure)

    @property
    def num_factor(self) -> int: return len(self.exposure.columns)

    @property
    def U(self) -> ndarray: return power(self.specific.values, 2)

    @property
    def F(self) -> ndarray: return self.exposure.values

    @property
    def X(self) -> ndarray: return self.factor_covar.values

    @property
    def R(self) -> ndarray:
        """

        """
        import warnings
        from numpy import diag
        warnings.warn(f'calling risk matrix directly consumes memory', category=UserWarning)
        ans = diag(self.U) + matmul(self.F, matmul(self.X, self.F.T))
        return ans

    @property
    def invU(self) -> ndarray: return 1. / self.U

    @property
    def invF(self) -> ndarray: return self.F * self.invU.reshape([-1, 1])

    @property
    def invX(self) -> ndarray:
        """

        """
        if not isinstance(self._inv_factor_covar, ndarray):
            # woodbury identity
            invF = self.invF
            kernel = matmul(self.F.T, invF) + invert(self.X)
            ans = -1 * invert(kernel)
            self._inv_factor_covar = ans
        return self._inv_factor_covar

    @property
    def index(self) -> Index: return self.exposure.index

    def __init__(self, exposure: DataFrame, factor_covar: DataFrame, specific: Series) -> None:
        """

        """
        assert exposure.index.equals(specific.index)
        assert factor_covar.index.equals(factor_covar.columns)
        assert allclose(factor_covar.values.T, factor_covar.values, rtol=0., atol=EPSILON)
        assert factor_covar.columns.equals(exposure.columns)

        self.exposure = exposure
        self.factor_covar = factor_covar
        self.specific = specific
        self._inv_factor_covar = None

    def align(self, index: Index, inplace: bool=False) -> 'FactorModel':
        """

        """
        if len(index.difference(self.index)) != 0:
            raise ValueError('new index contains extra asset ids')
        exposure = self.exposure.reindex(index=index)
        specific = self.specific.reindex(index=index)
        if inplace:
            self.exposure = exposure
            self.specific = specific
            self._inv_factor_covar = None
            ans = self
        else:
            ans = FactorModel(exposure=exposure, factor_covar=self.factor_covar, specific=specific)
        return ans

    def covariance(self, data: DataFrame) -> DataFrame:
        """
            measure the ex-ante covariance
        """
        if len(data.index.difference(self.index)) > 0:
            raise ValueError('input data contains asset out side the risk model coverage')
        if not data.index.equals(self.index):
            data = data.reindex(index=self.index, fill_value=0.)
        # compute idio and factor
        idio = data.values * self.U.reshape([-1, 1])
        idio = matmul(data.values.T, idio)
        factor = matmul(self.F.T, data.values)
        factor = matmul(factor.T, matmul(self.X, factor))
        # package ans
        ans = idio + factor
        ans = DataFrame(data=ans, index=data.columns, columns=data.columns)
        return ans

    def neutralize(self, label: DataFrame, feature: Optional[DataFrame]=None) -> DataFrame:
        """
            let feature be M, label be y
            solve \beta minimize
            \beta.T M.T R^-1 M \beta - 2 \beta.T M.T R^-1 y
        """
        if feature is None:
            feature = self.exposure
        assert feature.index.equals(self.index) and label.index.equals(self.index)

        # compute M.T R^-1 M first
        invU = self.invU
        invF = self.invF
        invMU = feature.values.T * invU.reshape([1, -1])
        invMF = matmul(feature.values.T, invF)
        invMUM = matmul(invMU, feature.values)
        kernel = matmul(invMF, self.invX)
        invMRM = invMUM + matmul(kernel, invMF.T)

        # compute M.T R^-1 y
        invMUy = matmul(invMU, label.values)
        invMRy = invMUy + matmul(kernel, matmul(invF.T, label.values))

        # now estimate beta
        beta = matmul(invert(invMRM), invMRy)
        explain = matmul(feature.values, beta)
        resid = label.values - explain

        # package ans
        columns = [f'net_{col}' for col in label.columns]
        ans = DataFrame(data=resid, index=label.index, columns=columns)
        return ans

    def apply_invert(self, data: DataFrame) -> DataFrame:
        """
            compute R^-1 \alpha
        """
        if not self.index.equals(data.index):
            raise ValueError('risk model is not aligned with the data')
        invF = self.invF
        invU = self.invU
        # compute idio and factor
        idio = data.values * invU.reshape([-1, 1])
        factor = matmul(invF.T, data.values)
        factor = matmul(invF, matmul(self.invX, factor))
        # package ans
        ans = idio + factor
        ans = DataFrame(data=ans, index=data.index, columns=data.columns)
        return ans
