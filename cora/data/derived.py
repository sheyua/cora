from typing import Dict, Optional
from pandas import DataFrame, date_range, concat
from ..util.epoch import Date
from ..util.multiprocessing import dispatch_with_pool
from .datafeed import DataFeed


class Derived(DataFeed):
    """
        Derived data feed split data daily for easy retrieval
    """
    @property
    def table_name(self) -> str: raise RuntimeError('should not hit this line')

    @property
    def converters(self) -> Dict[str, type]: raise RuntimeError('should not hit this line')

    def read_data(self, start_date: Date, end_date: Date) -> DataFrame: raise RuntimeError('should not hit this line')

    def read_with_date(self, dobj: Date, content: str) -> DataFrame: raise NotImplementedError

    def read_batch(self, start_date: Date, end_date: Date, content: str, num_worker: Optional[int]=None) -> DataFrame:
        """

        """
        inputs = date_range(start=start_date, end=end_date, freq=self.freq).tolist()
        inputs = [Date.convert(dobj=dobj) for dobj in inputs]
        if num_worker is None:
            num_worker = self.config.sub_root(section='State').as_int(key='num_worker', default=1)
        ans = dispatch_with_pool(num_worker=num_worker, inputs=inputs, target=self.read_with_date, state=self.state,
                                 context_method='spawn', content=content)
        ans = concat(ans, axis=0, sort=True)
        return ans
