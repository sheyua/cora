from numpy import diag, sqrt
from typing import List, Tuple
from pandas import DataFrame, Series, date_range, concat
from ..container.account import Account
from ..data.riskmodel import RiskModel
from ..data.begofday import BegOfDay
from ..data.endofday import EndOfDay
from ..stats.smoothing import WindowArrEWMA, compute_alpha
from ..util.epoch import Date
from ..util.config import parser
from ..util.compare import le
from ..util.constant import NAN
from ..util.multiprocessing import dispatch_with_pool
from .workflow import Workflow


class PortfolioEval(Workflow):
    """

    """
    def compute_markowitz_front(self, event_date: Date) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
            compute the cash neutral markowitz front and measure its risk
            1) universe is the intersection of trading universe and risk estimate universe
            2) signal values outside its funding universe are assumed to be zero
            3) markowitz front in SPO is defined as the solution to
                    min h.T R h
                    sub h.T A = b
                    where A = [\alpha, \ones] and b = [1, 0]
                        for unit alpha exposure and cash neutrality
                for simplicity, we make use only cash neutrality here where the solution is analytically available as
                    h \sim  R^-1 \alpha - (\ones.T R^-1 \alpha)/N \ones
        """
        # load risk model
        feed = RiskModel()
        model = feed.read_daily(event_date=event_date)

        # load signals available as of bod of event (trading) date
        feed = BegOfDay()
        meta = feed.read_daily(event_date=event_date, content='meta')
        universe = meta[meta['in_trading_universe'] == 'Y'].index
        universe = universe.intersection(model.index).sort_values()

        # loop through each signal feed and each signal to compute markowitz front
        target, variance = list(), list()
        for clipping, feed_name in zip(self.apply_clipping, self.signal_feed):
            data = feed.read_daily(event_date=event_date, content=feed_name)
            if clipping:
                self.logger.debug(f'clipping {feed_name} between {self.asset_clipping} and {1 - self.asset_clipping}')
                lower = data.quantile(self.asset_clipping)
                upper = data.quantile(1 - self.asset_clipping)
                data.clip(lower=lower, upper=upper, inplace=True, axis=1)
            data = data.reindex(index=universe, fill_value=0.)
            data.fillna(value=0., inplace=True)
            # trim the risk model to be on this universe
            model_use = model.align(index=universe, inplace=False)
            value = model_use.apply_invert(data=data)
            # the cash neutrality effective demean R^-1 \alpha
            value = value.sub(value.mean(axis=0), axis=1)
            # now measure the covariance
            covar = model_use.covariance(data=value)
            var = Series(data=diag(covar), index=covar.index, name='variance')
            # package for return
            target.append(value)
            variance.append(var)
            self.logger.info(f'computed {feed_name} target markowitz front and variance on {event_date}')
        # put all feed together, they all on the same index
        target = concat(target, axis=1, sort=True)
        variance = concat(variance, axis=0).to_frame()
        variance.index.name = 'signal_name'
        # attach date information
        target.reset_index(inplace=True)
        variance.reset_index(inplace=True)
        target['event_date'] = event_date.timestamp
        variance['event_date'] = event_date.timestamp
        target.set_index(['event_date', 'security_id'], inplace=True)
        variance.set_index(['event_date', 'signal_name'], inplace=True)
        variance = variance['variance'].unstack()
        variance.columns.name = None

        # also fetch realized returns to format accounting report
        feed = EndOfDay()
        label = feed.read_with_date(dobj=event_date, content='realize')
        columns = ['win_ret1d', 'spec_ret']
        return target, variance, label[columns]

    def target_risk_adjust(self, target: DataFrame, variance: DataFrame) -> DataFrame:
        """
            adjust the leverage of the target so that it tracks rolling 5% ex-ante risk
        """
        # first smooth the variance and compute variance scale factor
        alpha = compute_alpha(halflife=self.risk_rolling_halflife)
        obj_ = WindowArrEWMA(alpha=alpha, size=len(variance.columns), window=self.risk_rolling_window, infinite='raise')
        svar = variance.apply(obj_, axis=1, result_type='broadcast')
        svar[le(svar, 0.)] = NAN
        var_scale = self.target_annual_variance / svar
        var_scale.fillna(value=0., inplace=True)

        # then compute target scale factor
        ans = target.copy()
        target_scale = sqrt(var_scale)
        index = target.index.get_level_values('event_date')
        for col in ans.columns:
            ans[col] *= index.map(target_scale[col])
        return ans

    def smooth_trade(self, data: DataFrame) -> DataFrame:
        """
            further smooth fast target holding to represent trade cost
        """
        # first smooth the variance and compute variance scale factor
        alpha = compute_alpha(halflife=self.trade_smooth_halflife)
        obj_ = WindowArrEWMA(alpha=alpha, size=len(data.columns), window=self.trade_smooth_window, infinite='raise')
        ans = data.apply(obj_, axis=1, result_type='broadcast')
        return ans

    def run(self) -> None:
        """

        """
        inputs = date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        inputs = [Date.convert(dobj=dobj) for dobj in inputs]
        detail = dispatch_with_pool(num_worker=self.num_worker, inputs=inputs, target=self.compute_markowitz_front,
                                    state=self.state, context_method='spawn')
        # package inputs for making risk adjustment
        raw = concat([val for val, *_ in detail], axis=0, sort=True)
        variance = concat([val for _, val, _ in detail], axis=0, sort=True)
        # adjust target portfolio for rolling risk
        target = self.target_risk_adjust(target=raw, variance=variance)
        label = concat([val for *_, val in detail], axis=0, sort=True)

        # now make two accounting reports
        for col in target.columns:
            # fast target holding without smoothing
            fast = target[col].unstack(fill_value=0.)
            # slow target holding with smoothing
            slow = self.smooth_trade(data=fast)
            columns = slow.columns
            assert columns.equals(fast.columns)
            # fetch total and specific returns for this alpha
            total = label['win_ret1d'].unstack(fill_value=0.).reindex(columns=columns, fill_value=0.)
            specific = label['spec_ret'].unstack(fill_value=0.).reindex(columns=columns, fill_value=0.)
            # make an account
            dirname = self.dirname.format(signal_name=col, speed='fast')
            Account(target=fast, total=total, specific=specific).serialize(dirname=dirname)
            self.logger.info(f'fast target for {col} saved at {dirname}')
            dirname = self.dirname.format(signal_name=col, speed='slow')
            Account(target=slow, total=total, specific=specific).serialize(dirname=dirname)
            self.logger.info(f'slow target for {col} saved at {dirname}')

    @property
    def signal_feed(self) -> List[str]: return self.config.as_list(key='signal_feed')

    @property
    def apply_clipping(self) -> List[bool]:
        """

        """
        ans = self.config.as_list(key='apply_clipping')
        assert len(ans) == len(self.signal_feed)
        return [parser.as_bool(value=value) for value in ans]

    @property
    def asset_clipping(self) -> float: return self.config.as_float(key='asset_clipping', default=0.05)

    @property
    def target_annual_risk(self) -> float: return self.config.as_float(key='target_annual_risk', default=0.05)

    @property
    def target_annual_variance(self) -> float: return self.target_annual_risk ** 2

    @property
    def risk_rolling_window(self) -> int: return self.config.as_int(key='risk_rolling_window', default=252)

    @property
    def risk_rolling_halflife(self) -> int: return self.config.as_int(key='risk_rolling_halflife', default=126)

    @property
    def trade_smooth_window(self) -> int: return self.config.as_int(key='trade_smooth_window', default=5)

    @property
    def trade_smooth_halflife(self) -> int: return self.config.as_int(key='trade_smooth_halflife', default=2)

    @property
    def dirname(self) -> str: return self.config.as_path(key='dirname')
