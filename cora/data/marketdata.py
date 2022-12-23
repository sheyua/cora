from typing import Dict
from pandas import Series, DataFrame, to_datetime
from ..util.epoch import Date
from .datafeed import DataFeed


class MarketData(DataFeed):
    """

    """
    @property
    def table_name(self) -> str: return 'security_reference_data_w_ret1d.csv'

    @property
    def converters(self) -> Dict[str, type]:
        """

        """
        return {
            'data_date': str,
            'security_id': str,
            'close_price': float,
            'volume': int,
            'group_id': str,
            'in_trading_universe': str,
            'ret1d': float
        }

    def read_total_returns(self, start_date: Date, end_date: Date) -> Series:
        """

        """
        columns = ['data_date', 'security_id', 'ret1d']
        data = self.read_data(start_date=start_date, end_date=end_date)
        data.drop(columns=data.columns.difference(columns), inplace=True)
        data['data_date'] = to_datetime(data['data_date'], format=self.date_format)
        data.dropna(subset=['ret1d'], inplace=True)
        # columns
        index_columns = ['data_date', 'security_id']
        assert data.duplicated(subset=index_columns).sum() == 0
        data.set_index(index_columns, inplace=True)
        ans = data.squeeze()
        return ans

    def read_sector_exposure(self, start_date: Date, end_date: Date) -> DataFrame:
        """

        """
        columns = ['data_date', 'security_id', 'group_id']
        data = self.read_data(start_date=start_date, end_date=end_date)
        data.drop(columns=data.columns.difference(columns), inplace=True)
        data['data_date'] = to_datetime(data['data_date'], format=self.date_format)
        # limit to GICS2
        data['group_id'] = data['group_id'].str[:2]
        ans = data.groupby(columns)['group_id'].count().unstack()
        ans = ans.notnull().astype(float)
        ans.columns = [f'G{col}' for col in ans.columns]
        return ans
