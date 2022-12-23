from numpy import power
from pandas import DataFrame, Series, date_range, concat, merge
from ..data.marketdata import MarketData
from ..data.riskmodel import RiskModel
from ..data.begofday import BegOfDay
from ..data.endofday import EndOfDay
from ..util.epoch import Date
from ..util.multiprocessing import dispatch_with_pool
from .workflow import Workflow


class EstimateEval(Workflow):
    """

    """
    def compute_average_notional(self, end_date: Date) -> Series:
        """

        """
        content = dict()
        for shift in range(self.average_notional_window):
            data_date = end_date.roll(freq=self.freq, shift=-shift)
            try:
                feed = EndOfDay()
                data = feed.read_daily(data_date=data_date, content='realize')
                data = data['notional']
            except FileNotFoundError as err:
                self.logger.warning(f'eod notional for {data_date} is not available, fetch raw data: {err}')
                feed = MarketData()
                data = feed.read_data(start_date=data_date, end_date=data_date)
                assert data['security_id'].duplicated().sum() == 0
                data.set_index('security_id', inplace=True)
                data = data['close_price'] * data['volume']
            content[shift] = data
        content = DataFrame.from_dict(content, orient='columns')
        ans = content.mean(axis=1).rename(self.average_notional_label)
        return ans

    def calibrate(self, event_date: Date) -> None:
        """
            compute the following things
            (1) group_id, trading_universe use yesterday's EOD content (data_date)
            (2) trailing average notional
            (3) liquidity and risk adjusted regr weights
            and break down to daily files
        """
        data_date = event_date.roll(freq=self.freq, shift=-1)
        detail = list()

        # use yesterday's EOD content (data_date)
        feed = EndOfDay()
        columns = ['group_id', 'in_trading_universe']
        try:
            data = feed.read_daily(data_date=data_date, content='realize')
            data.drop(columns=data.columns.difference(columns), inplace=True)
        except FileNotFoundError as err:
            self.logger.warning(f'eod content for {data_date} is not available, fetch raw data: {err}')
            feed = MarketData()
            data = feed.read_data(start_date=data_date, end_date=data_date)
            assert data['security_id'].duplicated().sum() == 0
            data.set_index('security_id', inplace=True)
            data.drop(columns=data.columns.difference(columns), inplace=True)
        detail.append(data)

        # compute notional
        data = self.compute_average_notional(end_date=data_date)
        detail.append(data)
        detail = concat(detail, axis=1, sort=True)
        detail.dropna(axis=0, inplace=True)

        # fetch spec risk, compute liquidity weights
        feed = RiskModel()
        srisk = feed.read_specific(event_date=event_date)
        assert srisk.isnull().sum() == 0
        ans = merge(detail, srisk, left_index=True, right_index=True, how='inner')
        # regr weight is inverse variance
        wgts1 = power(ans['spec_risk'], 2)
        lower = wgts1.quantile(self.asset_variance_floor)
        wgts1 = 1. / wgts1.clip(lower=lower, upper=None)
        wgts2 = ans[self.average_notional_label] / ans[self.average_notional_label].mean()
        upper = wgts2.quantile(self.asset_notional_ceil)
        wgts2.clip(lower=0., upper=upper, inplace=True)
        wgts3 = power(wgts2, self.asset_notional_power) * wgts1
        ans['regr_wgts'] = wgts3
        self.logger.info(f'computed regr weights for {len(ans)} securities for trading on {event_date}')

        # (5) save this data
        feed = BegOfDay()
        feed.serialize(data=ans, content='meta', event_date=event_date)

    def run(self) -> None:
        """

        """
        inputs = date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        inputs = [Date.convert(dobj=dobj) for dobj in inputs]
        self.logger.info(f'running estimate data generation for {len(inputs)} dates on {self.num_worker} worker')
        dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.calibrate, state=self.state,
                           context_method='spawn')

    @property
    def average_notional_window(self) -> int: return self.config.as_int(key='average_notional_window', default=20)

    @property
    def average_notional_label(self) -> str: return f'adn{self.average_notional_window}'

    @property
    def asset_variance_floor(self) -> float: return self.config.as_float(key='asset_variance_floor', default=0.1)

    @property
    def asset_notional_ceil(self) -> float: return self.config.as_float(key='asset_notional_ceil', default=0.9)

    @property
    def asset_notional_power(self) -> float: return self.config.as_float(key='asset_notional_power', default=0.3)
