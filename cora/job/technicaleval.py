from numpy import sqrt, power
from pandas import DataFrame, date_range, merge
from ..data.riskmodel import RiskModel
from ..data.begofday import BegOfDay
from ..data.endofday import EndOfDay
from ..util.epoch import Date
from ..util.compare import gt
from ..util.multiprocessing import dispatch_with_pool
from .workflow import Workflow


class TechnicalEval(Workflow):
    """

    """
    def compute_impulse(self, event_date: Date) -> DataFrame:
        """

        """
        detail = dict()
        data_date = event_date.roll(freq=self.freq, shift=-1)

        # load past returns
        feed = EndOfDay()
        data = feed.read_daily(data_date=data_date, content='realize')
        data.drop(columns=data.columns.difference(['win_ret1d', 'spec_ret', 'notional']), inplace=True)
        # load meta data back then
        feed = BegOfDay()
        refer = feed.read_daily(event_date=event_date, content='meta')

        # first compute reversal
        label = [col for col in refer.columns if col.startswith('adn')].pop()
        value = merge(refer[label], data, left_index=True, right_index=True, how='inner')
        estimate = value[label].clip(lower=value[label].quantile(self.asset_clipping), upper=None)
        assert gt(estimate.min(), 0.)
        surp = sqrt(value['notional'] / estimate).clip(lower=1. / self.surprise_clipping, upper=self.surprise_clipping)
        value = -1 * value['spec_ret'] / surp
        # TODO add this #
        lower, upper = value.quantile(self.asset_clipping), value.quantile(1 - self.asset_clipping)
        value = value.clip(lower=lower, upper=upper)
        assert gt(value.std(), 0) and gt(refer['spec_risk'].median(), 0)
        value = value * refer['spec_risk'].median() / value.std()
        # TODO add this #
        detail['short_rev'] = value

        # second compute sub-industry momentum, we do not have market cap so use sqrt notional to weight
        columns = [label, 'group_id']
        value = merge(refer[columns], data['win_ret1d'], left_index=True, right_index=True, how='inner')
        # individual contribution
        upper = value[label].quantile(1 - self.asset_clipping)
        value['w_i'] = sqrt(value[label].clip(lower=0., upper=upper))
        value['r_i * w_i'] = value['win_ret1d'] * value['w_i']
        gb = value.groupby('group_id')
        value['\sum_j r_j * w_j'] = value['group_id'].map(gb['r_i * w_i'].sum())
        value['\sum_j w_j'] = value['group_id'].map(gb['w_i'].sum())
        value['\sum_j!=i r_j * w_j'] = value['\sum_j r_j * w_j'] - value['r_i * w_i']
        value['\sum_j!=i w_j'] = value['\sum_j w_j'] - value['w_i']
        lower = value['\sum_j!=i w_j'].quantile(self.asset_clipping)
        value['\sum_j!=i w_j'].clip(lower=lower, upper=None, inplace=True)
        assert gt(value['\sum_j!=i w_j'].min(), 0.)
        value = value['\sum_j!=i r_j * w_j'] / value['\sum_j!=i w_j']
        # this signal needs to be scale by volatility
        # TODO add this #
        lower, upper = value.quantile(self.asset_clipping), value.quantile(1 - self.asset_clipping)
        value = value.clip(lower=lower, upper=upper)
        value -= value.mean()
        assert gt(value.std(), 0.)
        value /= value.std()
        # TODO add this #
        value = value * refer['spec_risk'].reindex(index=value.index, fill_value=0.)
        detail['ind_mom'] = value

        # now format the output
        ans = DataFrame.from_dict(detail, orient='columns')
        ans.fillna(value=0., inplace=True)
        return ans

    def calibrate(self, event_date: Date) -> None:
        """
            compute the following things
            (1) simple reversal of specific returns, conditioned by volume surprise
            (2) sub-industry momentum
            and break down to daily files
        """
        ans = None
        for shift in range(self.smooth_window):
            remote_event_date = event_date.roll(freq=self.freq, shift=-shift)
            weight = power(0.5, shift / self.smooth_halflife)
            try:
                data = self.compute_impulse(event_date=remote_event_date) * weight
            except FileNotFoundError as err:
                self.logger.warning(f'pass compute on {remote_event_date}: {err}')
            else:
                if isinstance(ans, DataFrame):
                    index = ans.index.intersection(data.index)
                    data = data.reindex(index=index, fill_value=0.)
                    ans = ans.reindex(index=index, fill_value=0.).add(data, axis=0)
                else:
                    ans = data
        if not isinstance(ans, DataFrame):
            self.logger.warning(f'no valid impulse found on {event_date}, pad with zero')
            feed = RiskModel()
            srisk = feed.read_specific(event_date=event_date)
            ans = DataFrame(index=srisk.index, columns=['short_rev', 'ind_mom'], data=0.)
        else:
            # perform clipping and demeaning to make the signal cash neutral in its funding universe
            lower = ans.quantile(self.asset_clipping, axis=0)
            upper = ans.quantile(1 - self.asset_clipping, axis=0)
            ans.clip(lower=lower, upper=upper, inplace=True, axis=1)
            # demean
            ans = ans.sub(ans.mean(axis=0), axis=1)

        # cache signal
        assert isinstance(ans, DataFrame)
        feed = BegOfDay()
        feed.serialize(data=ans, content='signal/technical', event_date=event_date)

    def run(self) -> None:
        """

        """
        inputs = date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        inputs = [Date.convert(dobj=dobj) for dobj in inputs]
        self.logger.info(f'running technical signal generation for {len(inputs)} dates on {self.num_worker} worker')
        dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.calibrate, state=self.state,
                           context_method='spawn')

    @property
    def smooth_window(self) -> int: return self.config.as_int(key='smooth_window', default=5)

    @property
    def smooth_halflife(self) -> int: return self.config.as_int(key='smooth_halflife', default=2)

    @property
    def asset_clipping(self) -> float: return self.config.as_float(key='asset_clipping', default=0.01)

    @property
    def surprise_clipping(self) -> float: return self.config.as_float(key='surprise_clipping', default=2.)
