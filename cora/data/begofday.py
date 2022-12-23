from pandas import DataFrame, read_parquet
from ..util.epoch import Date
from ..util.file import mkdir, dirname
from .derived import Derived


class BegOfDay(Derived):
    """

    """
    def read_with_date(self, dobj: Date, content: str) -> DataFrame:
        """

        """
        ans = self.read_daily(event_date=dobj, content=content)
        assert ans.index.name == 'security_id'
        ans.reset_index(inplace=True)
        ans['event_date'] = dobj.timestamp
        ans.set_index(['event_date', 'security_id'], inplace=True)
        return ans

    def read_daily(self, event_date: Date, content: str) -> DataFrame:
        """
            can be used for trading as of bod of event_date
        """
        filename = self.filename.format(content=content, event_date=event_date.strftime(self.date_format))
        ans = read_parquet(filename)
        return ans

    def serialize(self, data: DataFrame, content: str, event_date: Date) -> None:
        """

        """
        # save data
        filename = self.filename.format(content=content, event_date=event_date.strftime(self.date_format))
        mkdir(path=dirname(filename))
        data.to_parquet(filename, compression='gzip')
        self.logger.info(f'saved {content} data for trading at bod of {event_date}')
