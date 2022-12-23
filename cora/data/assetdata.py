from typing import List, Dict
from pandas import DataFrame, to_datetime
from ..util.epoch import Date
from .datafeed import DataFeed


class AssetData(DataFeed):
    """

    """
    @property
    def table_name(self) -> str: return 'risk_factors.csv'

    @property
    def num_factor(self) -> int: return 6

    @property
    def factor_start_index(self) -> int: return 1

    @property
    def factor_name(self) -> List[str]: return [f'rf{idx + self.factor_start_index}' for idx in range(self.num_factor)]

    @property
    def converters(self) -> Dict[str, type]:
        """

        """
        ans = {
            'data_date': str,
            'security_id': str,
        }
        for col in self.factor_name:
            ans[col] = float
        return ans

    def read_style_exposure(self, start_date: Date, end_date: Date) -> DataFrame:
        """
            these factors are standardized already and slow moving
        """
        data = self.read_data(start_date=start_date, end_date=end_date)
        data['data_date'] = to_datetime(data['data_date'], format=self.date_format)
        # columns
        index_columns = ['data_date', 'security_id']
        assert data.duplicated(subset=index_columns).sum() == 0
        data.set_index(index_columns, inplace=True)
        for col in self.factor_name:
            data[col].fillna(value=0., inplace=True)
        return data
