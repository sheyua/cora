from typing import List
from numpy import ndarray, power, sqrt, array
from pandas import DataFrame, date_range, merge
from pandas.tseries.offsets import CustomBusinessMonthBegin
from ..data.marketdata import MarketData
from ..data.assetdata import AssetData
from ..data.riskmodel import RiskModel
from ..stats.outlier import adjust_outlier
from ..stats.linear import decompose_label
from ..stats.covariance import newey_west, max_entropy
from ..util.compare import le, gt
from ..util.constant import ANNUAL_FACTOR
from ..util.epoch import Date
from ..util.multiprocessing import dispatch_with_pool
from .workflow import Workflow


class RiskModelEval(Workflow):
    """

    """
    def get_weight(self, num_obs: int) -> ndarray:
        """

        """
        # create decay weight
        assert self.halflife > 0
        ans = power(.5, 1 / self.halflife)
        ans = power(ans, num_obs - 1 - array(range(num_obs)))
        ans = ans / ans.sum()
        return ans

    def __init__(self, name: str) -> None:
        """

        """
        super(RiskModelEval, self).__init__(name=name)
        self.calibrate_freq = CustomBusinessMonthBegin(calendar=self.calendar)

    def fetch_returns(self, event_date: Date) -> DataFrame:
        """
            aligned the with model traded as of beginning of event_date
        """
        data_date = event_date.roll(freq=self.freq, shift=-1)
        start_date = data_date.roll(freq=self.freq, shift=-self.train_window)
        # load returns from data_date - window to data_date
        feed = MarketData()
        data = feed.read_total_returns(start_date=start_date, end_date=data_date)
        ans = data.unstack()
        self.logger.info(f'fetch returns of {len(ans)} dates for {len(ans.columns)} assets for bod {event_date}')

        # first filter out those with too many observation missing
        notnull = ans.notnull().mean(axis=0)
        invalid = notnull[le(notnull, self.minimum_notnull)].index
        self.logger.info(f'{len(invalid)} assets do not have more than {self.minimum_notnull:.2f} valid '
                         f'returns for bod {event_date}')
        ans.drop(columns=invalid, inplace=True)

        # the returns will be adjusted both timeseries wise and cross sectional wise
        adjust_outlier(data=ans, na_threshold=self.na_threshold, clip_threshold=self.clip_threshold, inplace=True)
        return ans

    def fetch_factor(self, event_date: Date) -> DataFrame:
        """
            contains both the style factors and sector factors
        """
        data_date = event_date.roll(freq=self.freq, shift=-1)
        start_date = data_date.roll(freq=self.freq, shift=-self.train_window)
        # load style exposure from data_date - window to data_date
        feed = AssetData()
        style = feed.read_style_exposure(start_date=start_date, end_date=data_date)
        feed = MarketData()
        sector = feed.read_sector_exposure(start_date=start_date, end_date=data_date)
        # join the two components to obtain factors
        ans = merge(style, sector, left_index=True, right_index=True, how='inner')
        self.logger.info(f'fetch {len(ans.columns)} factors from {start_date} to {data_date} for bod {event_date}')
        assert ans.isnull().sum(axis=1).sum() == 0
        return ans

    def calibrate(self, event_date: Date) -> None:
        """

        """
        # (1) first fetch feature and label
        feature = self.fetch_factor(event_date=event_date)
        label = self.fetch_returns(event_date=event_date)
        # (2) first estimate the full sample asset total risk to weight the linear regression
        weight = label.var(axis=0, ddof=0)
        lower = weight.quantile(self.asset_variance_floor)
        assert gt(lower, 0.)
        weight.clip(lower=lower, upper=None, inplace=True)
        weight = 1. / weight
        # (3) estimate daily specific returns and factor returns
        beta, resid = dict(), dict()
        for dobj in label.index.intersection(feature.index.unique('data_date')).sort_values().to_list():
            value = decompose_label(feature=feature.loc[dobj], label=label.loc[dobj], weight=weight)
            beta[dobj], resid[dobj] = value
        self.logger.info(f'solve factor and specific returns for {len(beta)} dates')
        # this is the solved factor returns
        beta = DataFrame.from_dict(beta, orient='index')
        beta.index.name = 'data_date'
        resid = DataFrame.from_dict(resid, orient='index')
        resid.index.name = 'data_date'

        # (4) for factor returns, perform newey west, then shrink
        weight = self.get_weight(num_obs=len(beta))
        factor_covar = newey_west(data=beta.values, weight=weight, lag=self.autocorr_lag)
        factor_covar = max_entropy(data=factor_covar, prior=None) * ANNUAL_FACTOR
        factor_covar = DataFrame(data=factor_covar, index=beta.columns, columns=beta.columns)

        # (5) for spec returns, just compute the weighed variance
        weight_sum = resid.notnull().values * weight.reshape([-1, 1])
        weight_sum = weight_sum.sum(axis=0)
        assert not le(weight_sum, 0.).any()
        spec_var = power(resid, 2) * weight.reshape([-1, 1])
        spec_var = spec_var.sum(axis=0) * ANNUAL_FACTOR / weight_sum
        spec_risk = sqrt(spec_var)
        spec_risk.index.name = 'security_id'
        lower = spec_risk.quantile(self.asset_variance_floor)
        spec_risk.clip(lower=lower, upper=None, inplace=True)

        # (6) dump this model, also dump exposure to quick retrieval
        feed = RiskModel()
        feed.serialize(factor_covar=factor_covar, spec_risk=spec_risk, event_date=event_date)

    def run(self) -> None:
        """

        """

        inputs = self.calibrate_date()
        self.logger.info(f'running risk model calibration for {len(inputs)} dates on {self.num_worker} worker')
        dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.calibrate, state=self.state,
                           context_method='spawn')

    def calibrate_date(self) -> List[Date]:
        """

        """
        ans = date_range(start=self.start_date, end=self.end_date, freq=self.calibrate_freq)
        ans = [Date.convert(dobj=dobj) for dobj in ans]
        return ans

    @property
    def train_window(self) -> int: return self.config.as_int(key='train_window', default=756)

    @property
    def halflife(self) -> int: return self.config.as_int(key='halflife', default=252)

    @property
    def autocorr_lag(self) -> int: return self.config.as_int(key='autocorr_lag', default=2)

    @property
    def minimum_notnull(self) -> float: return self.config.as_float(key='minimum_notnull', default=0.95)

    @property
    def na_threshold(self) -> float: return self.config.as_float(key='na_threshold', default=10.)

    @property
    def clip_threshold(self) -> float: return self.config.as_float(key='clip_threshold', default=3.)

    @property
    def asset_variance_floor(self) -> float: return self.config.as_float(key='asset_variance_floor', default=0.1)
