from typing import List
from numpy import isfinite, power
from pandas import Series, DataFrame, date_range, merge, concat
from ..data.vendor import Vendor
from ..data.riskmodel import RiskModel
from ..data.begofday import BegOfDay
from ..util.epoch import Date
from ..util.compare import le
from ..util.constant import NAN, ANNUAL_FACTOR
from ..util.multiprocessing import dispatch_with_pool
from .workflow import Workflow


class FundamentalEval(Workflow):
    """

    """
    def compute(self, data: DataFrame, grouping: Series) -> Series:
        """
            compute the surprise against its post quarterly numbers
            then industry de- median and surprise
            finally add a drift term
        """
        # first compute the announcement surprise
        mean = data.mean(axis=0)
        stddev = data.std(axis=0, ddof=1)
        stddev[le(stddev, 0.)] = NAN
        surp = data.fillna(method='ffill', limit=self.data_window).tail(1).T.squeeze() - mean
        surp = surp / stddev
        surp = surp.dropna().clip(lower=-self.zscore_clipping, upper=self.zscore_clipping)
        # compute the industry demean version
        surp = merge(surp.rename('value'), grouping.str[:self.industry_group], left_index=True, right_index=True)
        group = surp.groupby('group_id')['value'].median()
        surp['group_median'] = surp['group_id'].map(group)
        surp['adj_value'] = surp['value'] - surp['group_median'].fillna(0.)
        ans = surp['adj_value'].rename('last')

        # now compute the post-announcement drift
        report = data.tail(self.drift_window).reindex(columns=ans.index)
        report.values[isfinite(report.values)] = 1.
        report = report.multiply(ans, axis=1)
        weight = power(0.5, 1 / self.drift_halflife)
        weight = power(weight, range(self.drift_window - 1, -1, -1))
        drift = report.multiply(weight, axis=0).sum(axis=0, min_count=1).rename('drift').dropna()

        # package ans
        ans = concat([ans, drift], axis=1, sort=True)
        return ans

    def calibrate(self, event_date: Date) -> None:
        """
            compute the industry relative surprise and make two constructs
            1) hold as a constant
            2) use it as a post-announcement drift
        """
        feed = RiskModel()
        srisk = feed.read_specific(event_date=event_date)
        feed = BegOfDay()
        grouping = feed.read_daily(event_date=event_date, content='meta')['group_id']
        end_date = event_date.roll(freq=self.freq, shift=-1)
        start_date = end_date.roll(freq=self.freq, shift=-self.data_window)

        detail = list()
        for vendor_id in self.include_data_set:
            feed = Vendor(vendor_id=vendor_id)
            data = feed.read_data(start_date=start_date, end_date=end_date)
            data = data.groupby(['data_date', 'security_id'])[feed.field].mean().unstack()
            data = self.compute(data=data, grouping=grouping)
            data.columns = [f'{col}{vendor_id}' for col in data.columns]
            if vendor_id in self.invert_sign:
                data = data * -1
            detail.append(data)
        ans = concat(detail, axis=1, sort=True)

        # align with srisk and scale by srisk
        index = ans.index.intersection(srisk.index).sort_values()
        ans = ans.reindex(index=index).multiply(srisk.reindex(index=index), axis=0)

        # cache signal
        assert isinstance(ans, DataFrame)
        feed = BegOfDay()
        feed.serialize(data=ans, content='signal/fundamental', event_date=event_date)

    def run(self) -> None:
        """

        """
        inputs = date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        inputs = [Date.convert(dobj=dobj) for dobj in inputs]
        self.logger.info(f'running fundamental signal generation for {len(inputs)} dates on {self.num_worker} worker')
        dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.calibrate, state=self.state,
                           context_method='spawn')

    @property
    def include_data_set(self) -> List[str]: return self.config.as_list(key='include_data_set')

    @property
    def invert_sign(self) -> List[str]: return self.config.as_list(key='invert_sign', default=[])

    @property
    def quarter_window(self) -> int: return self.config.as_int(key='quarter_window', default=4)

    @property
    def drift_window(self) -> int: return self.config.as_int(key='drift_window', default=5)

    @property
    def drift_halflife(self) -> int: return self.config.as_int(key='drift_halflife', default=2)

    @property
    def data_window(self) -> int: return int((self.quarter_window + 0.5) * ANNUAL_FACTOR / 4)

    @property
    def zscore_clipping(self) -> float: return self.config.as_float(key='zscore_clipping', default=2.)

    @property
    def industry_group(self) -> int: return 4
