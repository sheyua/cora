from logging import Logger
from pandas_market_calendars import get_calendar
from pandas.tseries.offsets import CustomBusinessDay
from ..core.state import State
from ..util.config import Config
from ..util.epoch import Date


class Workflow(object):
    """

    """
    def __init__(self, name: str) -> None:
        """

        """
        self.name = name
        self.calendar = get_calendar(name=self.composite_exchange, open_time=None, close_time=None)
        self.freq = CustomBusinessDay(holidays=self.calendar.adhoc_holidays, calendar=self.calendar.regular_holidays)

    def run(self) -> None: raise NotImplementedError

    @property
    def state(self) -> State: return State.get()

    @property
    def config(self) -> Config: return self.state.config.sub_root(section=self.name)

    @property
    def logger(self) -> Logger: return self.state.logger

    @property
    def num_worker(self) -> int:
        """

        """
        ans = self.config.sub_root(section='State').as_int(key='num_worker', default=1)
        return self.config.as_int(key='num_worker', default=ans)

    @property
    def start_date(self) -> Date:
        """

        """
        ans = self.state.config.sub_root(section='State').as_str(key='start_date')
        return Date.from_str(dobj=ans)

    @property
    def end_date(self) -> Date:
        """

        """
        ans = self.state.config.sub_root(section='State').as_str(key='end_date')
        return Date.from_str(dobj=ans)

    @property
    def composite_exchange(self) -> str:
        """
            under-lining data shows this is NYSE calendar
        """
        return self.state.config.sub_root(section='State').as_str(key='composite_exchange')
