from pandas import DataFrame, read_parquet
from ..util.epoch import Date
from ..util.file import mkdir, dirname
from .derived import Derived


class EndOfDay(Derived):
    """

    """
    def read_with_date(self, dobj: Date, content: str) -> DataFrame:
        """

        """
        ans = self.read_daily(data_date=dobj, content=content)
        assert ans.index.name == 'security_id'
        ans.reset_index(inplace=True)
        ans['data_date'] = dobj.timestamp
        ans.set_index(['data_date', 'security_id'], inplace=True)
        return ans

    def read_daily(self, data_date: Date, content: str) -> DataFrame:
        """
            can be read as of end of data_date
        """
        filename = self.filename.format(content=content, data_date=data_date.strftime(self.date_format))
        ans = read_parquet(filename)
        return ans

    def serialize(self, data: DataFrame, content: str, data_date: Date) -> None:
        """

        """
        # save data
        filename = self.filename.format(content=content, data_date=data_date.strftime(self.date_format))
        mkdir(path=dirname(filename))
        data.to_parquet(filename, compression='gzip')
        self.logger.info(f'saved {content} data at eod of {data_date}')
