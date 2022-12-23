from typing import List
from pandas import Series, DataFrame, date_range
from ..data.vendor import Vendor
from ..data.riskmodel import RiskModel
from ..data.begofday import BegOfDay
from ..util.epoch import Date
from ..util.compare import le
from ..util.constant import NAN
from ..util.multiprocessing import dispatch_with_pool
from .workflow import Workflow


class ZScoreEval(Workflow):
    """

    """
    def compute(self, data: DataFrame) -> Series:
        """

        """
        mean = data.mean(axis=0)
        stddev = data.std(axis=0, ddof=1)
        stddev[le(stddev, 0.)] = NAN
        ans = data.tail(n=1).T.squeeze() - mean
        ans = ans / stddev
        return ans.dropna().clip(lower=-self.zscore_clipping, upper=self.zscore_clipping)

    def calibrate(self, event_date: Date) -> None:
        """

        """
        feed = RiskModel()
        srisk = feed.read_specific(event_date=event_date)
        end_date = event_date.roll(freq=self.freq, shift=-1)
        start_date = end_date.roll(freq=self.freq, shift=-self.zscore_window)

        detail = dict()
        for vendor_id in self.include_data_set:
            feed = Vendor(vendor_id=vendor_id)
            data = feed.read_data(start_date=start_date, end_date=end_date)
            data = data.groupby(['data_date', 'security_id'])[feed.field].mean().unstack()
            data = self.compute(data=data)
            if vendor_id in self.invert_sign:
                data = data * -1
            detail[f'z{vendor_id}'] = data
        ans = DataFrame.from_dict(detail, orient='columns')

        # align with srisk and scale by srisk
        index = ans.index.intersection(srisk.index).sort_values()
        ans = ans.reindex(index=index).multiply(srisk.reindex(index=index), axis=0)

        # cache signal
        assert isinstance(ans, DataFrame)
        feed = BegOfDay()
        feed.serialize(data=ans, content='signal/zscore', event_date=event_date)

    def run(self) -> None:
        """

        """
        inputs = date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        inputs = [Date.convert(dobj=dobj) for dobj in inputs]
        self.logger.info(f'running zscore signal generation for {len(inputs)} dates on {self.num_worker} worker')
        dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.calibrate, state=self.state,
                           context_method='spawn')

    @property
    def include_data_set(self) -> List[str]: return self.config.as_list(key='include_data_set')

    @property
    def invert_sign(self) -> List[str]: return self.config.as_list(key='invert_sign', default=[])

    @property
    def zscore_window(self) -> int: return self.config.as_int(key='zscore_window', default=10)

    @property
    def zscore_clipping(self) -> float: return self.config.as_float(key='zscore_clipping', default=2.)
