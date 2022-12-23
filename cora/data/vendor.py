from typing import Dict
from pandas import DataFrame
from ..util.epoch import Date
from .datafeed import DataFeed


class Vendor(DataFeed):
    """

    """
    @property
    def can_early_stop(self) -> bool: return self._can_early_stop

    def __init__(self, vendor_id: str) -> None:
        """

        """
        super(Vendor, self).__init__()
        self.vendor_id = vendor_id
        self._can_early_stop = self.vendor_id in ['1', '2', '5', '8', '9', '10']
        # 3, 4, 6, 7, 11 are not sorted

    @property
    def table_name(self) -> str: return f'data_set_{self.vendor_id}.csv'

    @property
    def field(self) -> str: return f'd{self.vendor_id}'

    @property
    def converters(self) -> Dict[str, type]:
        """

        """
        return {
            'data_date': str,
            'security_id': str,
            self.field: float,
        }

    def read_data(self, start_date: Date, end_date: Date) -> DataFrame:
        """
            the files are not necessarily sorted
        """
        ans = super(Vendor, self).read_data(start_date=start_date, end_date=end_date)
        ans.sort_values(by=['data_date', 'security_id'], ascending=True, inplace=True)
        return ans
