from numpy import power
from typing import List, Tuple, Dict
from pandas import Series, DataFrame, date_range, concat
from ..container.datahub import DataHub
from ..data.riskmodel import RiskModel
from ..data.trademodel import TradeModel
from ..data.endofday import EndOfDay
from ..data.begofday import BegOfDay
from ..stats.booster import cross_validate, train_with_hold
from ..util.constant import ANNUAL_FACTOR
from ..util.config.parser import JSONType
from ..util.epoch import Date
from ..util.multiprocessing import dispatch_with_pool, chunk
from .workflow import Workflow


class TradeModelEval(Workflow):
    """

    """
    def fetch_label(self, event_date: Date) -> Series:
        """
            event_date is BOD, trades are place on EOD
            forward return starts to realize on the second day
        """
        ans, feed = list(), EndOfDay()
        # roll forward one day, EOD of start_date holdings are formed
        start_date = event_date.roll(freq=self.freq, shift=1)
        for shift in range(self.forward_horizon):
            data_date = start_date.roll(freq=self.freq, shift=shift)
            data = feed.read_daily(data_date=data_date, content='realize')['spec_ret']
            ans.append(data)
        ans = concat(ans, axis=1, sort=True)
        ans = ans.sum(axis=1)
        ans.name = f'spec_ret_{self.forward_horizon}d'
        return ans

    def fetch_feature(self, event_date: Date) -> DataFrame:
        """
            feature on the join universe of features
        """
        ans, feed = list(), BegOfDay()
        for name in self.signal_feed:
            data = feed.read_daily(event_date=event_date, content=name)
            # data.columns = [f'{name}.{col}' for col in data.columns]
            ans.append(data)
        ans = concat(ans, axis=1, sort=True)
        return ans

    def fetch_weight(self, event_date: Date, shift: int) -> Series:
        """
            use liquidity weight, also decay given the model halflife
        """
        feed = BegOfDay()
        decay = power(.5, shift / self.train_halflife_day)
        ans = feed.read_daily(event_date=event_date, content='meta')['regr_wgts']
        return ans * decay

    def fetch_data(self, inputs: List[Tuple[int, Date]]) -> DataHub:
        """

        """
        ans = None
        for shift, event_date in inputs:
            # fetch three sources and align index
            label = self.fetch_label(event_date=event_date)
            feature = self.fetch_feature(event_date=event_date)
            weight = self.fetch_weight(event_date=event_date, shift=shift)
            specific = RiskModel().read_specific(event_date=event_date)
            if ans is None:
                ans = DataHub(feature_column=feature.columns.tolist(), label_name=str(label.name))
            ans.append(feature=feature, label=label, weight=weight, specific=specific)
        return ans

    def calibrate(self, event_date: Date) -> None:
        """
            model is prepared as of bod convention ready to be trade after BOD of event_date
        """
        # first determine the start and end date to fetch data
        # cannot load forward return after this
        end_date = event_date.roll(freq=self.freq, shift=-self.blackout)
        start_date = end_date.roll(freq=self.freq, shift=-self.train_window_day)
        start_date = max(self.record_start, start_date)
        # get the list of days from start to end
        inputs = date_range(start=start_date, end=end_date, freq=self.freq).to_list()
        num_date = len(inputs)
        inputs = [(num_date - index, Date.convert(dobj=dobj)) for index, dobj in enumerate(inputs)]
        if len(inputs) <= self.min_num_date:
            self.logger.info(f'calibrate date {event_date} load data from {start_date} to {end_date}')
        else:
            # reduce the amount of resource used in the machine
            num_extra = len(inputs) - self.min_num_date
            num_needed = int(num_extra * (1 - self.down_sample_rate)) + self.min_num_date
            self.logger.info(f'calibrate date {event_date} sample {num_needed} dates from {start_date} to {end_date}')
            inputs = chunk.chunkize_sequential(inputs=inputs, num_worker=num_needed)
            inputs = [value for value, *_ in inputs]
            assert len(inputs) == num_needed

        # fetch training data, first train a linear model, then train a non-linear model
        data = self.fetch_data(inputs=inputs)
        prior = self.linear_prior_adjust
        for col in data.feature_column:
            prior[col] = prior.get(col, 1.)
        linear = data.compute_linear(shrinkage=self.linear_shrinkage, prior=prior)
        # then we train a nonlinear model
        self.logger.info(f'start calibrating nonlinear model on {event_date}')
        x_data, y_data, weight = data.exfoliate_non_linear(linear=linear['beta'].values)
        params = cross_validate(search_params=self.search_params, x_data=x_data, y_data=y_data, weight=weight,
                                cv=3, override=None)
        self.logger.info(f'cv best params on {event_date} is {params}')
        model = train_with_hold(params=params, x_data=x_data, y_data=y_data, weight=weight, hold_size=.3,
                                plot_metric=False)

        # from numpy import save
        # from ..util.file import mkdir
        # folder = '/home/yuan/Downloads/test/big'
        # mkdir(folder)
        # save(f'{folder}/x_data', x_data)
        # save(f'{folder}/y_data', y_data)
        # save(f'{folder}/weight', weight)

        # finally put together the trade model and save
        feed = TradeModel()
        feed.serialize(linear=linear, booster=model, event_date=event_date)

    def run(self) -> None:
        """

        """

        inputs = self.calibrate_date()
        self.logger.info(f'running trade model calibration for {len(inputs)} dates on {self.num_worker} worker')
        dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.calibrate, state=self.state,
                           context_method='spawn')

    def calibrate_date(self) -> List[Date]:
        """
            annually calibrated to save computes
        """
        ans = list()
        for dobj in date_range(start=self.start_date, end=self.end_date, freq=self.freq).to_list():
            dobj = Date.convert(dobj=dobj)
            if dobj >= Date(year=dobj.year, month=6, day=30):
                dobj = Date(year=dobj.year, month=6, day=30).roll(freq=self.freq, shift=1)
            else:
                dobj = Date(year=dobj.year, month=1, day=1).roll(freq=self.freq, shift=1)
            dobj = min(max(dobj, self.start_date), self.end_date)
            if dobj not in ans:
                ans.append(dobj)
        return ans

    @property
    def linear_shrinkage(self) -> float: return self.config.as_float(key='linear_shrinkage')

    @property
    def linear_prior_adjust(self) -> Dict[str, float]:
        """

        """
        ans = dict()
        with self.config.sub(section='linear_prior_adjust') as sub:
            for key in sub.scalars:
                ans[key] = sub.as_float(key=key)
        return ans

    @property
    def signal_feed(self) -> List[str]: return self.config.as_list(key='signal_feed')

    @property
    def forward_horizon(self) -> int: return self.config.as_int(key='forward_horizon', default=5)

    @property
    def train_window_year(self) -> int: return self.config.as_int(key='train_window_year', default=10)

    @property
    def train_halflife_year(self) -> int: return self.config.as_int(key='train_halflife_year', default=5)

    @property
    def min_num_date(self) -> int: return self.config.as_int(key='min_num_date', default=252 * 10)

    @property
    def down_sample_rate(self) -> float: return self.config.as_float(key='down_sample_rate', default=0.)

    @property
    def train_window_day(self) -> int: return self.train_window_year * ANNUAL_FACTOR

    @property
    def train_halflife_day(self) -> int: return self.train_halflife_year * ANNUAL_FACTOR

    @property
    def blackout(self) -> int: return self.forward_horizon + 2

    @property
    def record_start(self) -> Date: return RiskModel().record_start

    # --- LGBMRegressor settings --- #
    @property
    def search_params(self) -> JSONType: return self.config['search_params']
    # --- LGBMRegressor settings --- #
