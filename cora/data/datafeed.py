from io import StringIO
from logging import Logger
from typing import List, Dict
from pandas import DataFrame, read_csv
from pandas.tseries.offsets import CustomBusinessDay
from pandas_market_calendars import get_calendar
from zipfile import ZipFile
from ..core.state import State
from ..util.config import Config
from ..util.constant import NAN
from ..util.epoch import Date


class DataFeed(object):
    """
        location reserved in the yaml
    """
    @property
    def config(self) -> Config: return self.state.config.sub(section=f'DataFeed/{self.__class__.__name__}')

    @property
    def logger(self) -> Logger: return self.state.logger

    @property
    def state(self) -> State: return State.get()

    @property
    def filename(self) -> str: return self.config.as_path(key='filename')

    @property
    def table_name(self) -> str: raise NotImplementedError

    @property
    def encoding(self) -> str: return 'utf-8'

    @property
    def date_format(self) -> str: return '%Y%m%d'

    @property
    def usecols(self) -> List[str]: return sorted(self.converters.keys())

    @property
    def converters(self) -> Dict[str, type]: raise NotImplementedError

    @property
    def can_early_stop(self) -> bool: return True

    @property
    def composite_exchange(self) -> str:
        """
            under-lining data shows this is NYSE calendar
        """
        return self.state.config.sub_root(section='State').as_str(key='composite_exchange')

    def __init__(self) -> None:
        """

        """
        self.calendar = get_calendar(name=self.composite_exchange, open_time=None, close_time=None)
        self.freq = CustomBusinessDay(holidays=self.calendar.adhoc_holidays, calendar=self.calendar.regular_holidays)

    def read_data(self, start_date: Date, end_date: Date) -> DataFrame:
        """
            all dates are data date
        """
        start_bytes = start_date.strftime(self.date_format).encode(self.encoding)
        end_bytes = end_date.strftime(self.date_format).encode(self.encoding)
        size = len(start_bytes)

        # csv contains 'NA' and int volume as 10+6
        use_converters = dict()
        for col, dtype in self.converters.items():
            if issubclass(dtype, (int, float)):
                use_converters[col] = str
            else:
                use_converters[col] = dtype

        # read only part of the information
        with ZipFile(self.filename, 'r') as zf:
            file_ = zf.open(name=self.table_name)
            detail = list()
            header = file_.readline().decode(self.encoding).rstrip()
            detail.append(header)
            for line in file_.readlines():
                date_bytes = line[:size]
                if date_bytes < start_bytes:
                    continue
                elif date_bytes > end_bytes:
                    if self.can_early_stop:
                        # earlier stop because the data is sorted
                        break
                    else:
                        continue
                else:
                    detail.append(line.decode(self.encoding).rstrip())
            io_ = StringIO('\n'.join(detail))
            ans = read_csv(io_, sep=',', usecols=self.usecols, converters=use_converters)

        # put the type conversion back
        for col, dtype in self.converters.items():
            if issubclass(dtype, int):
                ans[col] = ans[col].astype(float).astype(int)
            elif issubclass(dtype, float):
                mask = ans[col].isin(['', 'NA'])
                ans.loc[mask, col] = NAN
                ans[col] = ans[col].astype(float)
        return ans
