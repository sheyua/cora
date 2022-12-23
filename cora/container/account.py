from typing import Optional, Tuple, Any
from numpy import sqrt, power
from pandas import DataFrame, Series, read_parquet, concat
from ..util.compare import le
from ..util.epoch import Date
from ..util.file import mkdir
from ..util.constant import ANNUAL_FACTOR, INF


def compute_drawdown(data: Series) -> Tuple[float, int]:
    """
        data is the returns series
    """
    data = data.cumsum()
    high_watermark = -INF
    drawdown, duration = 0., 0
    max_drawdown, max_duration = 0., 0
    for value in data.to_list():
        if le(high_watermark, value):
            # update high watermark
            high_watermark = value
            # reset drawdown and duration
            drawdown, duration = 0., 0
        else:
            drawdown = min(drawdown, value - high_watermark)
            duration += 1
        # update max duration and max drawdown
        max_duration = max(max_duration, duration)
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown, max_duration


class Account(object):
    """

    """
    @staticmethod
    def retrieve(signal_name: str, speed: str='slow', section: str='PortfolioEval',
                 readjust_leverage: bool=False, since: Optional[Date]=None, end: Optional[Date]=None) -> 'Account':
        """
            for notebook plotting usage
        """
        from ..core.state import State
        State.console_state(filename='../yaml/data.yaml', stdout='ERROR', stderr='ERROR')
        with State.get().config.sub(section=section) as sub:
            dirname = sub.as_path(key='dirname').format(signal_name=signal_name, speed=speed)
        return Account.deserialize(dirname=dirname, readjust_leverage=readjust_leverage, since=since, end=end)

    @staticmethod
    def deserialize(dirname: str, readjust_leverage: bool=False, since: Optional[Date]=None,
                    end: Optional[Date]=None) -> 'Account':
        """

        """
        target = read_parquet(f'{dirname}/target.parquet.gz')
        total = read_parquet(f'{dirname}/total.parquet.gz')
        try:
            specific = read_parquet(f'{dirname}/specific.parquet.gz')
        except (FileNotFoundError, OSError):
            specific = None
        if since is not None:
            target = target.loc[since.timestamp:]
            total = total.loc[since.timestamp:]
            if specific is not None:
                specific = specific.loc[since.timestamp:]
        if end is not None:
            target = target.loc[:end.timestamp]
            total = total.loc[:end.timestamp]
            if specific is not None:
                specific = specific.loc[:end.timestamp]
        if readjust_leverage:
            from ..util.compare import le
            from ..util.constant import INF
            long = target.clip(lower=0, upper=None)
            short = target.clip(lower=None, upper=0)
            long_size, short_size = long.sum(axis=1), short.abs().sum(axis=1)
            # avoid zero division
            long_size[le(long_size, 0.)] = INF
            short_size[le(long_size, 0.)] = INF
            long = long.divide(long_size, axis=0)
            short = short.divide(short_size, axis=0)
            target = long + short
        return Account(target=target, total=total, specific=specific)

    def serialize(self, dirname: str) -> None:
        """

        """
        mkdir(dirname)
        self.target.to_parquet(f'{dirname}/target.parquet.gz', compression='gzip')
        self.total.to_parquet(f'{dirname}/total.parquet.gz', compression='gzip')
        if isinstance(self.specific, DataFrame):
            self.specific.to_parquet(f'{dirname}/specific.parquet.gz', compression='gzip')

    def __init__(self, target: DataFrame, total: DataFrame, specific: Optional[DataFrame]=None) -> None:
        """

        """
        assert target.index.equals(total.index)
        if isinstance(specific, DataFrame):
            assert specific.index.equals(target.index)
        columns = target.columns.union(total.columns)
        if isinstance(specific, DataFrame):
            columns = columns.union(specific.columns)
        columns = columns.sort_values()
        # pass in target holdings
        target = target.reindex(columns=columns, fill_value=0.)
        target.fillna(value=0., inplace=True)
        self.target = target
        # pass in total returns
        total = total.reindex(columns=columns, fill_value=0.)
        total.fillna(value=0., inplace=True)
        self.total = total
        if isinstance(specific, DataFrame):
            specific = specific.reindex(columns=columns, fill_value=0.)
            specific.fillna(value=0., inplace=True)
            self.specific = specific
        else:
            self.specific = None

    def compute_gross(self, target: DataFrame, skip_specific: bool=False) -> DataFrame:
        """
            target realize tomorrow returns
        """
        holdings = target.shift(1).fillna(value=0.)
        total = holdings.multiply(self.total, axis=0).sum(axis=1)
        ans = total.to_frame(name='total')
        ans.index.name = 'data_date'
        if isinstance(self.specific, DataFrame) and not skip_specific:
            ans['specific'] = holdings.multiply(self.specific, axis=0).sum(axis=1)
        return ans

    def lead_lag_ir(self, lead: int, lag: int) -> Series:
        """

        """
        if lead >= 0 or lag <= 0:
            raise ValueError(f'bad lead {lead} lag {lag}')
        ans = dict()
        for shift in range(lead, lag + 1):
            target = self.target.shift(shift)
            gross = self.compute_gross(target=target, skip_specific=True)
            value = gross['total']
            value = sqrt(ANNUAL_FACTOR) * value.mean() / value.std()
            ans[shift] = value
        ans = Series(ans, name='lead_lag_ir')
        ans.index.name = 'shift'
        return ans

    @property
    def leverage(self) -> Series:
        """

        """
        ans = self.target.abs().sum(axis=1)
        ans.fillna(value=0., inplace=True)
        ans.index.name = 'data_date'
        return ans

    @property
    def gross(self) -> DataFrame: return self.compute_gross(target=self.target, skip_specific=False)

    @property
    def trade(self) -> DataFrame:
        """
            holding_t + 1 = target_t = holding_t * (1 + ret_t) + delta_t
        """
        holdings = self.target.shift(1).fillna(value=0.)
        holdings = holdings.multiply(1 + self.total, axis=0)
        ans = self.target.sub(holdings, axis=0)
        ans.index.name = 'data_date'
        return ans

    @property
    def turnover(self) -> Series: return self.trade.abs().sum(axis=1)

    @property
    def trade_cost(self) -> Series: return 1e-4 * self.turnover

    @property
    def net(self) -> DataFrame: return self.gross.sub(self.trade_cost, axis=0)

    @property
    def concentration(self) -> Series:
        """

        """
        holdings = self.target.shift(1).fillna(value=0.)
        leverage = holdings.abs().sum(axis=1)
        leverage[le(leverage, 0.)] = 1
        holdings = holdings.divide(leverage, axis=0)
        ans = power(holdings, 2).sum(axis=1)
        return ans

    @property
    def extreme_position(self) -> DataFrame:
        """

        """
        holdings = self.target.shift(1).fillna(value=0.)
        leverage = holdings.abs().sum(axis=1)
        leverage[le(leverage, 0.)] = 1
        min_position = holdings.min(axis=1) / leverage
        max_position = holdings.max(axis=1) / leverage
        ans = concat([min_position.rename('min'), max_position.rename('max')], axis=1)
        return ans

    def summary(self, fmt: bool=False) -> Series:
        """

        """
        concentration = self.concentration
        leverage = self.leverage
        turnover = self.turnover
        cost = self.trade_cost
        slippage = cost.sum() / turnover.sum()
        period = leverage.mean() * 2 / turnover.mean()
        universe = 1 / concentration.mean()

        gross = self.gross['total']
        net = self.net['total']
        # some annual number
        annual_cost = cost.mean() * ANNUAL_FACTOR
        annual_gross = gross.mean() * ANNUAL_FACTOR
        annual_gross_gmv = annual_gross / leverage.mean()
        annual_gross_risk = sqrt(ANNUAL_FACTOR) * gross.std(ddof=0)
        annual_gross_risk_gmv = annual_gross_risk / leverage.mean()
        annual_gross_ir = annual_gross_gmv / annual_gross_risk_gmv
        annual_net = net.mean() * ANNUAL_FACTOR
        annual_net_gmv = annual_net / leverage.mean()
        annual_net_risk = sqrt(ANNUAL_FACTOR) * net.std(ddof=0)
        annual_net_risk_gmv = annual_net_risk / leverage.mean()
        annual_net_ir = annual_net_gmv / annual_net_risk_gmv
        # drawdown
        max_drawdown, max_duration = compute_drawdown(data=net)
        max_drawdown_gmv = max_drawdown / leverage.mean()

        # put together the ans
        ans = {
            'AUM': '1 $' if fmt else 1.,
            'Gross Exposure': f'{leverage.mean():.2f} $' if fmt else leverage.mean(),
            'Concentration Implied Universe Size': f'{int(universe)}' if fmt else universe,
            'Daily Notional Trade': f'{turnover.mean():.2f} $' if fmt else turnover.mean(),
            'Holding Period': f'{period:.2f} D' if fmt else period,
            'Slippage on Close': f'{slippage * 1e4:.1f} bps' if fmt else slippage,
            'Annual Trading Cost': f'{annual_cost:.2f} $' if fmt else annual_cost,

            'Annual Gross Profit': f'{annual_gross:.2f} $' if fmt else annual_gross,
            'Annual Net Profit': f'{annual_net:.2f} $' if fmt else annual_net,
            'Annual Gross Return on GMV': f'{annual_gross_gmv * 100:.2f} %' if fmt else annual_gross_gmv,
            'Annual Net Return on GMV': f'{annual_net_gmv * 100:.2f} %' if fmt else annual_net_gmv,

            'Annual Gross Risk on AUM': f'{annual_gross_risk * 100:.2f} %' if fmt else annual_gross_risk,
            'Annual Net Risk on AUM': f'{annual_net_risk * 100:.2f} %' if fmt else annual_net_risk,
            'Annual Gross Risk on GMV': f'{annual_gross_risk_gmv * 100:.2f} %' if fmt else annual_gross_risk_gmv,
            'Annual Net Risk on GMV': f'{annual_net_risk_gmv * 100:.2f} %' if fmt else annual_net_risk_gmv,
            'Gross IR': f'{annual_gross_ir:.2f}' if fmt else annual_gross_ir,
            'Net IR': f'{annual_net_ir:.2f}' if fmt else annual_net_ir,

            'Worst Drawdown': f'{max_drawdown:.2f} $' if fmt else max_drawdown,
            'Worst Drawdown on GMV': f'{max_drawdown_gmv * 100:.2f} %' if fmt else max_drawdown_gmv,
            'Worst Drawdown Duration': f'{max_duration} D' if fmt else max_duration,
        }
        ans = Series(ans, name='summary')
        ans.index.name = 'metric'
        return ans

    def exhibit(self, **kwargs: Any) -> None:
        """

        """
        from matplotlib import pyplot as plt
        fig_width = kwargs.get('fig_width', 8)
        fig_height = kwargs.get('fig_height', 4)
        label_size = kwargs.get('label_size', 12)
        gross_line_alpha = kwargs.get('gross_line_alpha', 0.5)
        lead_lag = kwargs.get('lead_lag', (-10, 10))
        vbar_width = kwargs.get('vbar_width', 0.5)
        vbar_ylim = kwargs.get('vbar_ylim', None)

        if kwargs.get('plot_style', 'ggplot'):
            plt.style.use(kwargs.get('plot_style', 'ggplot'))

        if kwargs.get('display_summary', True):
            from IPython.display import HTML, display
            html = self.summary(fmt=True).to_frame().to_html()
            display(HTML(html))

        if kwargs.get('plot_leverage', True):
            fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
            # plot leverage on the left
            ax1.plot(self.leverage, label='leverage', color='red')
            ax1.set_xlabel('date', fontsize=label_size)
            ax1.set_ylabel('Leverage', fontsize=label_size, color='red')
            ax1.tick_params(axis='both', labelsize=label_size)
            # plot turnover on the right
            ax2 = plt.twinx()
            ax2.plot(self.turnover / self.leverage.mean() * 100, label='turnover', color='blue')
            ax2.set_ylabel('Turnover [% on GMV]', fontsize=label_size, color='blue')
            ax2.tick_params(axis='both', labelsize=label_size)
            fig.suptitle('Leverage and Turnover', fontsize=label_size)

        if kwargs.get('plot_timeseries', True):
            net = self.net.cumsum()
            gross = self.gross.cumsum()
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.plot(net['total'], label='net-total', color='red', linewidth=3)
            ax.plot(gross['total'], label='gross-total', color='red', linestyle='--', alpha=gross_line_alpha)
            if isinstance(self.specific, DataFrame):
                ax.plot(gross['specific'], label='gross-specific', color='blue', linestyle='--', alpha=gross_line_alpha)
                ax.plot(net['specific'], label='net-specific', color='blue')
            # set label and legends
            ax.set_xlabel('date', fontsize=label_size)
            ax.set_ylabel('Profit on AUM', fontsize=label_size)
            ax.tick_params(axis='both', labelsize=label_size)
            ax.legend(loc='best', fontsize=label_size)
            fig.suptitle('PnL Timeseries', fontsize=label_size)

        if kwargs.get('plot_lead_lag', True):
            lead, lag = lead_lag
            data = self.lead_lag_ir(lead=lead, lag=lag)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.bar(x=data.index, height=data, width=vbar_width)
            if vbar_ylim:
                ax.set_ylim(vbar_ylim)
            # set label and legends
            ax.set_xlabel('lag', fontsize=label_size)
            ax.set_ylabel('Lead Lag IR', fontsize=label_size)
            ax.tick_params(axis='both', labelsize=label_size)
            fig.suptitle('Lead Lag IR', fontsize=label_size)
