from typing import List
from pandas import DataFrame, date_range, concat
from ..data.trademodel import TradeModel
from ..data.riskmodel import RiskModel
from ..data.begofday import BegOfDay
from ..util.epoch import Date
from ..util.multiprocessing import dispatch_with_pool
from .workflow import Workflow


class TradeModelApply(Workflow):
    """

    """
    def calibrate(self, event_date: Date) -> None:
        """

        """
        # load model
        feed = TradeModel()
        model = feed.read_daily(event_date=event_date)
        model.non_linear_shrinkage = self.non_linear_shrinkage

        # load specific risk
        feed = RiskModel()
        srisk = feed.read_specific(event_date=event_date)

        # load feature
        feature, feed = list(), BegOfDay()
        for name in self.signal_feed:
            data = feed.read_daily(event_date=event_date, content=name)
            # data.columns = [f'{name}.{col}' for col in data.columns]
            feature.append(data)
        feature = concat(feature, axis=1, sort=True)
        ans = model(feature=feature, specific=srisk)

        # cache signal
        assert isinstance(ans, DataFrame)
        feed = BegOfDay()
        feed.serialize(data=ans, content='alpha', event_date=event_date)

    def run(self) -> None:
        """

        """
        inputs = date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        inputs = [Date.convert(dobj=dobj) for dobj in inputs]
        self.logger.info(f'running final alpha signal generation for {len(inputs)} dates on {self.num_worker} worker')
        dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.calibrate, state=self.state,
                           context_method='spawn')

    @property
    def non_linear_shrinkage(self) -> float: return self.config.as_float(key='non_linear_shrinkage')

    @property
    def signal_feed(self) -> List[str]:
        """

        """
        eval_section = self.config.as_str(key='eval_section')
        ans = self.config.sub_root(section=eval_section).as_list(key='signal_feed')
        return ans
