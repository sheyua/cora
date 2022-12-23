from typing import List
from numpy import power, log
from pandas import DataFrame, Series, date_range
from ..data.riskmodel import RiskModel
from ..data.marketdata import MarketData
from ..data.begofday import BegOfDay
from ..data.endofday import EndOfDay
from ..data.vendor import Vendor
from ..util.epoch import Date
from ..util.compare import gt, le
from ..util.constant import NAN
from ..util.multiprocessing import dispatch_with_pool
from .workflow import Workflow


class FlowEval(Workflow):
    """

    """
    def compute_smooth_imbalance(self, event_date: Date, vendor_id: str) -> Series:
        """
            trade imbalance smooth over window with halflife
        """
        data_date = event_date.roll(freq=self.freq, shift=-1)
        start_date = data_date.roll(freq=self.freq, shift=1 - self.smooth_window)
        feed = Vendor(vendor_id=vendor_id)
        data = feed.read_data(start_date=start_date, end_date=data_date)
        data = data.groupby(['data_date', 'security_id'])[feed.field].mean().unstack()
        if len(data) != self.smooth_window:
            index = date_range(start=start_date, end=data_date, freq=self.freq)
            str_index = [dobj.strftime(feed.date_format) for dobj in index]
            str_found = data.index.to_list()
            for missing in set(str_index).difference(str_found):
                self.logger.warning(f'data date {missing} is missing for vendor {vendor_id}')
                data.loc[missing, :] = 0.
            data = data.reindex(index=str_index)

        if vendor_id == '8':
            # d8 is quoted as volume
            # need past volume information to scale the signal
            volume, dobj = dict(), start_date
            while dobj <= data_date:
                try:
                    feed = EndOfDay()
                    value = feed.read_daily(data_date=dobj, content='realize')['volume']
                except FileNotFoundError as err:
                    self.logger.warning(f'eod volume data not available get raw: {err}')
                    feed = MarketData()
                    value = feed.read_data(start_date=dobj, end_date=dobj)
                    value = value.groupby('security_id')['volume'].mean()
                value[le(value, 0.)] = NAN
                volume[dobj] = value
                dobj = dobj.roll(freq=self.freq, shift=1)
            volume = DataFrame.from_dict(volume, orient='index')
            volume = volume.reindex(columns=data.columns, fill_value=NAN)
            assert len(volume) == self.smooth_window and volume.index.is_monotonic_increasing
            # just simply divide data by volume
            data.values[:] = data.values / volume.values
        elif vendor_id == '9':
            # d9 is quoted as percentage
            data = log(data.clip(lower=0, upper=100) + 1)
        else:
            raise NotImplementedError

        # asset level clipping before demeaning
        lower, upper = data.quantile(self.asset_clipping, axis=1), data.quantile(1 - self.asset_clipping, axis=1)
        data.clip(lower=lower, upper=upper, axis=0, inplace=True)
        mean, stddev = data.mean(axis=1), data.std(axis=1, ddof=0)
        assert gt(stddev, 0.).all()
        data = data.sub(mean, axis=0).divide(stddev, axis=0)
        data.clip(lower=-self.zscore_clipping, upper=self.zscore_clipping, inplace=True)

        # now smooth with decay
        weight = power(0.5, 1 / self.smooth_halflife)
        weight = power(weight, range(self.smooth_window - 1, -1, -1))
        weight = weight / weight.sum()
        ans = data.values * weight.reshape([-1, 1])
        ans = Series(data=ans.sum(axis=0), index=data.columns, name=f'f{vendor_id}').dropna()
        if vendor_id in self.invert_sign:
            ans = ans * -1
        return ans

    def calibrate(self, event_date: Date) -> None:
        """

        """
        ans = dict()
        for vendor_id in self.include_data_set:
            ans[f'f{vendor_id}'] = self.compute_smooth_imbalance(event_date=event_date, vendor_id=vendor_id)
        ans = DataFrame.from_dict(ans, orient='columns')
        # flow signals should always be scaled by volatility
        feed = RiskModel()
        srisk = feed.read_specific(event_date=event_date)
        ans = ans.sub(ans.mean(axis=0), axis=1)
        ans = ans.multiply(srisk.reindex(index=ans.index, fill_value=0.), axis=0)

        # cache signal
        assert isinstance(ans, DataFrame)
        feed = BegOfDay()
        feed.serialize(data=ans, content='signal/flow', event_date=event_date)

    def run(self) -> None:
        """

        """
        inputs = date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        inputs = [Date.convert(dobj=dobj) for dobj in inputs]
        self.logger.info(f'running flow signal generation for {len(inputs)} dates on {self.num_worker} worker')
        dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.calibrate, state=self.state,
                           context_method='spawn')

    @property
    def smooth_window(self) -> int: return self.config.as_int(key='smooth_window', default=5)

    @property
    def smooth_halflife(self) -> int: return self.config.as_int(key='smooth_halflife', default=2)

    @property
    def asset_clipping(self) -> float: return self.config.as_float(key='asset_clipping', default=0.01)

    @property
    def include_data_set(self) -> List[str]: return self.config.as_list(key='include_data_set')

    @property
    def invert_sign(self) -> List[str]: return self.config.as_list(key='invert_sign', default=[])

    @property
    def zscore_clipping(self) -> float: return self.config.as_float(key='zscore_clipping', default=2.)
