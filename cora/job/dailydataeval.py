from numpy import maximum, sqrt
from pandas import date_range
from ..data.marketdata import MarketData
from ..data.riskmodel import RiskModel
from ..data.endofday import EndOfDay
from ..util.constant import ANNUAL_FACTOR
from ..util.epoch import Date
from ..util.multiprocessing import dispatch_with_pool
from .workflow import Workflow


class DailyDataEval(Workflow):
    """

    """
    def calibrate(self, data_date: Date) -> None:
        """
            compute the following things
            (1) clipped total returns
            (2) specific returns
            (3) trading notional
            (4) keep group_id, trading_universe
            and break down to daily files
        """
        # (1) get the risk model
        feed = RiskModel()
        model = feed.read_daily(event_date=data_date)

        # (2) get raw returns and trading volume
        feed = MarketData()
        data = feed.read_data(start_date=data_date, end_date=data_date)
        assert data['data_date'].unique().tolist() == [data_date.strftime(feed.date_format)]
        data.drop(columns=['data_date'], inplace=True)
        assert data.duplicated(subset=['security_id']).sum() == 0
        data.set_index('security_id', inplace=True)

        # (3) trim the data to be only within risk estimate universe
        index = model.index.intersection(data.index).sort_values()
        data = data.reindex(index=index)
        model.align(index=index, inplace=True)

        # (4) generate the data items
        limit = sqrt(1 / ANNUAL_FACTOR) * model.specific * self.clip_total_return
        lower = maximum(self.total_return_lower, -1 * limit)
        data['win_ret1d'] = data['ret1d'].fillna(0.).clip(lower=lower, upper=limit)
        data['spec_ret'] = model.neutralize(label=data['win_ret1d'].to_frame()).squeeze()
        data['notional'] = data['close_price'] * data['volume']

        # (5) save this data
        feed = EndOfDay()
        feed.serialize(data=data, content='realize', data_date=data_date)

    def run(self) -> None:
        """

        """
        inputs = date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        inputs = [Date.convert(dobj=dobj) for dobj in inputs]
        self.logger.info(f'running daily data generation for {len(inputs)} dates on {self.num_worker} worker')
        dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.calibrate, state=self.state,
                           context_method='spawn')

    @property
    def clip_total_return(self) -> float: return self.config.as_float(key='clip_total_return', default=30.)

    @property
    def total_return_lower(self) -> float: return self.config.as_float(key='total_return_lower', default=-0.8)
